//internal/auth/service.go

package auth

import (
	"crypto/rand"
	"encoding/base32"
	"errors"
	"fmt"
	"net/http"
	"net/mail"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"defense-app/internal/models"
	"github.com/go-chi/jwtauth/v5"
	"github.com/pquerna/otp/totp"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"
)

// Service handles all authentication logic.
type Service struct {
	db        *gorm.DB
	tokenAuth *jwtauth.JWTAuth
}

func NewService(db *gorm.DB, tokenAuth *jwtauth.JWTAuth) *Service {
	return &Service{db: db, tokenAuth: tokenAuth}
}

// ── Registration ─────────────────────────────────────────────

type RegisterInput struct {
	Username string `json:"username" validate:"required,min=3,max=32"`
	Password string `json:"password" validate:"required,min=12"`
	Email    string `json:"email"    validate:"required,email"`
}

// ValidationError is returned when registration input is rejected.
// Handlers map it to HTTP 400 (instead of 409 "already taken").
type ValidationError struct{ Msg string }

func (e *ValidationError) Error() string { return e.Msg }

// Username: letters, digits, underscores, 3-32 chars. Case is preserved for
// display, but uniqueness and login matching are case-insensitive (see
// findByUsername below), so "Hema" and "hema" are the same account either way.
var usernameRe = regexp.MustCompile(`^[A-Za-z0-9_]{3,32}$`)

func validateRegister(inp *RegisterInput) error {
	inp.Username = strings.TrimSpace(inp.Username)
	inp.Email = strings.ToLower(strings.TrimSpace(inp.Email))

	if !usernameRe.MatchString(inp.Username) {
		return &ValidationError{"Username must be 3-32 characters: letters, numbers, and underscores only"}
	}
	addr, err := mail.ParseAddress(inp.Email)
	if err != nil || addr.Address != inp.Email {
		return &ValidationError{"Please enter a valid email address"}
	}
	if utf8.RuneCountInString(inp.Password) < 12 {
		return &ValidationError{"Password must be at least 12 characters"}
	}
	if len(inp.Password) > 72 { // bcrypt hard limit
		return &ValidationError{"Password must be at most 72 bytes"}
	}
	return nil
}

func (s *Service) Register(inp RegisterInput) (*models.User, error) {
	if err := validateRegister(&inp); err != nil {
		return nil, err
	}

	// Case-insensitive uniqueness: "Hema" and "hema" are the same account.
	var taken int64
	s.db.Model(&models.User{}).
		Where("lower(username) = lower(?) OR email = ?", inp.Username, inp.Email).
		Count(&taken)
	if taken > 0 {
		return nil, fmt.Errorf("username or email already taken")
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(inp.Password), bcrypt.DefaultCost)
	if err != nil {
		return nil, err
	}
	u := &models.User{
		Username:     inp.Username,
		Email:        inp.Email,
		PasswordHash: string(hash),
		Role:         "receiver", // default role; admin promotes as needed
		TOTPEnabled:  false,
	}
	if err := s.db.Create(u).Error; err != nil {
		return nil, fmt.Errorf("username or email already taken")
	}
	return u, nil
}

// findByUsername matches exactly first, then case-insensitively, so accounts
// created before this change (e.g. "hema") still work when typed as "Hema".
func (s *Service) findByUsername(name string, u *models.User) error {
	name = strings.TrimSpace(name)
	if err := s.db.Where("username = ?", name).First(u).Error; err == nil {
		return nil
	}
	return s.db.Where("lower(username) = lower(?)", name).Order("id ASC").First(u).Error
}

// ── Login (phase 1: credentials) ─────────────────────────────

type LoginResult struct {
	User        *models.User
	NeedsTOTP   bool
	PendingToken string // short-lived, only valid for TOTP phase
	SessionToken string // long-lived, only set when TOTP not required
}

func (s *Service) Login(username, password, ip, ua string) (*LoginResult, error) {
	var u models.User
	if err := s.findByUsername(username, &u); err != nil {
		return nil, errors.New("invalid credentials")
	}
	if err := bcrypt.CompareHashAndPassword([]byte(u.PasswordHash), []byte(password)); err != nil {
		return nil, errors.New("invalid credentials")
	}
	if !u.Active {
		return nil, errors.New("account disabled")
	}

	res := &LoginResult{User: &u, NeedsTOTP: u.TOTPEnabled}

	if u.TOTPEnabled {
		// Issue a short-lived pending token (5 min) — not a full session
		_, tokenStr, _ := s.tokenAuth.Encode(map[string]interface{}{
			"sub":     fmt.Sprint(u.ID),
			"role":    u.Role,
			"pending": true, // signals TOTP not yet confirmed
			"exp":     time.Now().Add(5 * time.Minute).Unix(),
		})
		res.PendingToken = tokenStr
	} else {
		tok, err := s.createSession(&u, ip, ua)
		if err != nil {
			return nil, err
		}
		res.SessionToken = tok
	}
	return res, nil
}

// ── Login (phase 2: TOTP) ─────────────────────────────────────

func (s *Service) VerifyTOTP(pendingToken, code, ip, ua string) (string, error) {
	token, err := s.tokenAuth.Decode(pendingToken)
	if err != nil {
		return "", errors.New("invalid pending token")
	}

	// 1. Safely extract and check the "pending" claim
	// Pass a pointer to the boolean where jwx will store the result
	var isPending bool
	if err := token.Get("pending", &isPending); err != nil || !isPending {
		return "", errors.New("not a pending token")
	}

	// 2. Safely extract the "sub" (user ID) claim
	// JSON numbers decode to float64, so we read it as a float first
	// 2. Safely extract the "sub" (user ID) claim as a STRING now
	var uidStr string
	if err := token.Get("sub", &uidStr); err != nil {
		return "", errors.New("missing or invalid user id in token")
	}
	
	// Convert the string back to a uint for GORM
	parsedID, err := strconv.ParseUint(uidStr, 10, 32)
	if err != nil {
		return "", errors.New("malformed user id in token")
	}
	uid := uint(parsedID)

	// 3. Find the user and validate the TOTP code
	var u models.User
	if err := s.db.First(&u, uid).Error; err != nil {
		return "", errors.New("user not found")
	}
	if !totp.Validate(code, u.TOTPSecret) {
		return "", errors.New("invalid TOTP code")
	}

	return s.createSession(&u, ip, ua)
}

// ── TOTP Setup ────────────────────────────────────────────────

type TOTPSetupResult struct {
	Secret  string
	QRURL   string
}

func (s *Service) SetupTOTP(userID uint) (*TOTPSetupResult, error) {
	var u models.User
	if err := s.db.First(&u, userID).Error; err != nil {
		return nil, err
	}
	key, err := totp.Generate(totp.GenerateOpts{
		Issuer:      "Defense Lab",
		AccountName: u.Email,
	})
	if err != nil {
		return nil, err
	}
	// Store secret but don't enable until user confirms with a valid code
	s.db.Model(&u).Update("totp_secret_pending", key.Secret())
	return &TOTPSetupResult{
		Secret: key.Secret(),
		QRURL:  key.URL(),
	}, nil
}

func (s *Service) ConfirmTOTP(userID uint, code string) error {
	var u models.User
	if err := s.db.First(&u, userID).Error; err != nil {
		return err
	}
	if !totp.Validate(code, u.TOTPSecretPending) {
		return errors.New("invalid code")
	}
	return s.db.Model(&u).Updates(map[string]interface{}{
		"totp_secret":         u.TOTPSecretPending,
		"totp_secret_pending": "",
		"totp_enabled":        true,
	}).Error
}

// ── Session management ────────────────────────────────────────

func (s *Service) createSession(u *models.User, ip, ua string) (string, error) {
	sessionID := randomHex(16)
	exp := time.Now().Add(8 * time.Hour)

	sess := models.Session{
		UserID:    u.ID,
		SessionID: sessionID,
		IP:        ip,
		UserAgent: ua,
		ExpiresAt: exp,
	}
	if err := s.db.Create(&sess).Error; err != nil {
		return "", err
	}

	_, tokenStr, err := s.tokenAuth.Encode(map[string]interface{}{
		"sub":        fmt.Sprint(u.ID),
		"role":       u.Role,
		"session_id": sessionID,
		"exp":        exp.Unix(),
	})
	return tokenStr, err
}

func (s *Service) RevokeSession(sessionID string) error {
	return s.db.Where("session_id = ?", sessionID).Delete(&models.Session{}).Error
}

func (s *Service) IsSessionValid(sessionID string) bool {
	var sess models.Session
	err := s.db.Where("session_id = ? AND expires_at > ?", sessionID, time.Now()).First(&sess).Error
	return err == nil
}

// ── Helpers ───────────────────────────────────────────────────

func randomHex(n int) string {
	b := make([]byte, n)
	rand.Read(b)
	return base32.StdEncoding.EncodeToString(b)[:n*2]
}

// ExtractUserFromRequest pulls the validated user out of the request context.
func ExtractUserFromRequest(r *http.Request) *models.User {
	u, _ := r.Context().Value(models.UserCtxKey).(*models.User)
	return u
}