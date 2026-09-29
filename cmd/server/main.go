//cmd/server/main.go

package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"defense-app/internal/auth"
	"defense-app/internal/handlers"
	"defense-app/internal/middleware"
	"defense-app/internal/models"
	"defense-app/internal/vault"

	"github.com/go-chi/chi/v5"
	chiMiddleware "github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	"github.com/go-chi/jwtauth/v5"
)

func main() {
	// ── Config from env ───────────────────────────────────────
	jwtSecret := getEnv("JWT_SECRET", "change-me-in-production-32chars!!")
	dbDSN := getEnv("DATABASE_URL", "postgres://defense:defense@localhost:5432/defense?sslmode=disable")
	port := getEnv("PORT", "8080")
	pythonURL := getEnv("PYTHON_SERVICE_URL", "http://localhost:5001")

	// ── Bootstrap ─────────────────────────────────────────────
	db, err := models.NewDB(dbDSN)
	if err != nil {
		log.Fatalf("db connect: %v", err)
	}
	if err := models.Migrate(db); err != nil {
		log.Fatalf("migrate: %v", err)
	}

	tokenAuth := jwtauth.New("HS256", []byte(jwtSecret), nil)
	authSvc := auth.NewService(db, tokenAuth)
	kv := vault.NewKeyVault(db)
	auditLog := models.NewAuditLogger(db)

	h := handlers.New(db, authSvc, kv, auditLog, tokenAuth, pythonURL)

	// ── Router ────────────────────────────────────────────────
	r := chi.NewRouter()

	// Global middleware
	r.Use(chiMiddleware.RequestID)
	r.Use(chiMiddleware.RealIP)
	r.Use(chiMiddleware.Logger)
	r.Use(chiMiddleware.Recoverer)
	r.Use(chiMiddleware.Timeout(120 * time.Second))
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-Request-ID"},
		AllowCredentials: true,
		MaxAge:           300,
	}))
	r.Use(middleware.RateLimiter(100, time.Minute))   // 100 req/min global
	r.Use(middleware.SecurityHeaders)

	// ── Public routes ─────────────────────────────────────────
	r.Group(func(r chi.Router) {
		r.Post("/api/auth/register", h.Register)
		r.Post("/api/auth/login", h.Login)
		r.Post("/api/auth/totp/verify", h.VerifyTOTP)
		r.Get("/api/health", h.Health)
	})

	// ── Authenticated routes ───────────────────────────────────
	r.Group(func(r chi.Router) {
		r.Use(jwtauth.Verifier(tokenAuth))
		r.Use(jwtauth.Authenticator(tokenAuth))
		r.Use(middleware.LoadUser(db))

		// Auth
		r.Post("/api/auth/logout", h.Logout)
		r.Get("/api/auth/me", h.Me)
		r.Post("/api/auth/become-sender", h.BecomeSender)
		r.Post("/api/auth/totp/setup", h.SetupTOTP)
		r.Get("/api/auth/sessions", h.ListSessions)
		r.Delete("/api/auth/sessions/{id}", h.RevokeSession)

		// All authenticated users — own job history (filtered by owner_id in handler)
		r.Get("/api/stego/jobs", h.ListJobs)
		r.Get("/api/stego/jobs/{id}", h.GetJob)

		// Sender — hide images
		r.Group(func(r chi.Router) {
			r.Use(middleware.RequireRole("sender", "admin"))
			r.Use(middleware.RateLimiter(10, time.Minute)) // stricter for ML ops
			r.Post("/api/stego/conceal", h.Conceal)        // upload cover + secrets
			// r.Get("/api/stego/jobs", h.ListJobs)
			// r.Get("/api/stego/jobs/{id}", h.GetJob)
			r.Get("/api/stego/jobs/{id}/stego", h.DownloadStego)
		})

		// Receiver — reveal images
		r.Group(func(r chi.Router) {
			r.Use(middleware.RequireRole("receiver", "admin"))
			r.Use(middleware.RateLimiter(10, time.Minute))
			r.Post("/api/stego/reveal", h.Reveal) // submit stego + key_id
			r.Get("/api/stego/reveal/{id}/secrets", h.DownloadSecrets)
		})

		// Key management — sender transfers key to receiver
		r.Group(func(r chi.Router) {
			r.Use(middleware.RequireRole("sender", "admin"))
			r.Post("/api/keys/{job_id}/transfer", h.TransferKey) // share key_id to receiver
		})
		r.Group(func(r chi.Router) {
			r.Use(middleware.RequireRole("receiver", "admin"))
			r.Get("/api/keys/inbox", h.KeyInbox) // keys sent to this receiver
		})

		// Admin only
		r.Group(func(r chi.Router) {
			r.Use(middleware.RequireRole("admin"))
			r.Get("/api/admin/users", h.ListUsers)
			r.Post("/api/admin/users", h.CreateUser)
			r.Put("/api/admin/users/{id}/role", h.UpdateUserRole)
			r.Delete("/api/admin/users/{id}", h.DeleteUser)
			r.Get("/api/admin/audit", h.AuditLog)
			r.Post("/api/admin/keys/rotate", h.RotateMasterKey)
		})
	})

	// ── Static frontend ───────────────────────────────────────
	r.Handle("/*", http.FileServer(http.Dir("./frontend/dist")))
	// r.Handle("/tmp/defense_outputs/*", http.StripPrefix("/tmp/defense_outputs/", http.FileServer(http.Dir("/tmp/defense_outputs"))))
	log.Printf("Defense server listening on :%s", port)
	if err := http.ListenAndServe(":"+port, r); err != nil {
		log.Fatal(err)
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}