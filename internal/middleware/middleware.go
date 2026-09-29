//internal/middleware/middleware.go

package middleware

import (
	"context"
	"net/http"
	"sync"
	"time"
	"strconv"

	"defense-app/internal/models"
	"github.com/go-chi/jwtauth/v5"
	"golang.org/x/time/rate"
	"gorm.io/gorm"
)

// ── Security headers ──────────────────────────────────────────

func SecurityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-XSS-Protection", "1; mode=block")
		w.Header().Set("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;")
		w.Header().Set("Referrer-Policy", "no-referrer")
		next.ServeHTTP(w, r)
	})
}

// ── User loader ───────────────────────────────────────────────

// LoadUser reads the JWT claims and attaches the full User to the context.
func LoadUser(db *gorm.DB) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, claims, err := jwtauth.FromContext(r.Context())
			if err != nil || claims == nil {
				http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
				return
			}

			// Reject pending tokens (TOTP not yet completed)
			if pending, _ := claims["pending"].(bool); pending {
				http.Error(w, `{"error":"totp required"}`, http.StatusUnauthorized)
				return
			}

			// 1. Extract the ID as a string
			uidStr, ok := claims["sub"].(string)
			if !ok {
				http.Error(w, `{"error":"invalid token format"}`, http.StatusUnauthorized)
				return
			}

			// 2. Parse the string back into an integer
			parsedID, err := strconv.ParseUint(uidStr, 10, 32)
			if err != nil {
				http.Error(w, `{"error":"malformed user id"}`, http.StatusUnauthorized)
				return
			}

			var u models.User
			if err := db.First(&u, uint(parsedID)).Error; err != nil || !u.Active {
				http.Error(w, `{"error":"user not found or disabled"}`, http.StatusUnauthorized)
				return
			}

			ctx := context.WithValue(r.Context(), models.UserCtxKey, &u)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// ── RBAC ──────────────────────────────────────────────────────

// RequireRole allows access only to users with one of the given roles.
func RequireRole(roles ...string) func(http.Handler) http.Handler {
	allowed := make(map[string]bool, len(roles))
	for _, r := range roles {
		allowed[r] = true
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			u, _ := r.Context().Value(models.UserCtxKey).(*models.User)
			if u == nil || !allowed[u.Role] {
				http.Error(w, `{"error":"forbidden"}`, http.StatusForbidden)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// ── Per-IP rate limiter ───────────────────────────────────────

type ipLimiter struct {
	mu       sync.Mutex
	limiters map[string]*rateLimiterEntry
	rate     rate.Limit
	burst    int
}

type rateLimiterEntry struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

var globalLimiter *ipLimiter

func RateLimiter(reqPerWindow int, window time.Duration) func(http.Handler) http.Handler {
	r := rate.Limit(float64(reqPerWindow) / window.Seconds())
	lim := &ipLimiter{
		limiters: make(map[string]*rateLimiterEntry),
		rate:     r,
		burst:    reqPerWindow,
	}
	// Cleanup stale entries every 5 minutes
	go func() {
		for range time.Tick(5 * time.Minute) {
			lim.mu.Lock()
			for ip, e := range lim.limiters {
				if time.Since(e.lastSeen) > 10*time.Minute {
					delete(lim.limiters, ip)
				}
			}
			lim.mu.Unlock()
		}
	}()

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ip := r.RemoteAddr
			lim.mu.Lock()
			e, ok := lim.limiters[ip]
			if !ok {
				e = &rateLimiterEntry{limiter: rate.NewLimiter(lim.rate, lim.burst)}
				lim.limiters[ip] = e
			}
			e.lastSeen = time.Now()
			l := e.limiter
			lim.mu.Unlock()

			if !l.Allow() {
				w.Header().Set("Retry-After", "60")
				http.Error(w, `{"error":"rate limit exceeded"}`, http.StatusTooManyRequests)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}