//internal/models/models.go

package models

import (
	"context"
	"log"
	"time"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// ── Context key ───────────────────────────────────────────────

type ctxKey string

const UserCtxKey ctxKey = "user"

// ── Models ────────────────────────────────────────────────────

type User struct {
	ID                 uint      `gorm:"primarykey"                json:"id"`
	Username           string    `gorm:"uniqueIndex;not null"      json:"username"`
	Email              string    `gorm:"uniqueIndex;not null"      json:"email"`
	PasswordHash       string    `gorm:"not null"                  json:"-"`
	Role               string    `gorm:"not null;default:'receiver'" json:"role"` // sender|receiver|admin
	Active             bool      `gorm:"not null;default:true"     json:"active"`
	TOTPEnabled        bool      `gorm:"not null;default:false"    json:"totp_enabled"`
	TOTPSecret         string    `gorm:"default:''"               json:"-"`
	TOTPSecretPending  string    `gorm:"default:''"               json:"-"`
	CreatedAt          time.Time `json:"created_at"`
	UpdatedAt          time.Time `json:"-"`
}

// Session tracks active JWT sessions for revocation.
type Session struct {
	ID        uint      `gorm:"primarykey" json:"id"`
	UserID    uint      `gorm:"not null;index" json:"user_id"`
	SessionID string    `gorm:"uniqueIndex;not null" json:"session_id"`
	IP        string    `json:"ip"`
	UserAgent string    `json:"user_agent"`
	ExpiresAt time.Time `json:"expires_at"`
	CreatedAt time.Time `json:"created_at"`
}

// StegoJob records each conceal/reveal operation.
type StegoJob struct {
	ID           string    `gorm:"primarykey" json:"id"` // UUID
	OwnerID      uint      `gorm:"not null;index" json:"owner_id"`
	Operation    string    `gorm:"not null" json:"operation"` // conceal|reveal
	Status       string    `gorm:"not null;default:'pending'" json:"status"` // pending|done|failed
	NumSecrets   int       `json:"num_secrets"`
	StegoPath    string    `json:"-"`   // server-side path, not exposed
	TextureScore float64   `json:"texture_score"`
	ErrorMsg     string    `json:"error,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"-"`
}

// KeyRecord stores the encrypted r_list (private reveal key).
type KeyRecord struct {
	ID            string    `gorm:"primarykey" json:"id"` // UUID
	JobID         string    `gorm:"not null;index" json:"job_id"`
	OwnerID       uint      `gorm:"not null" json:"owner_id"`
	EncryptedKey  []byte    `gorm:"not null" json:"-"` // AES-256-GCM encrypted r_list
	TransferredTo uint      `gorm:"default:0" json:"transferred_to,omitempty"`
	TransferredAt *time.Time `json:"transferred_at,omitempty"`
	CreatedAt     time.Time `json:"created_at"`
}

// AuditEntry is an append-only audit log.
type AuditEntry struct {
	ID        uint      `gorm:"primarykey"`
	UserID    uint      `gorm:"index"`
	Action    string    // LOGIN|CONCEAL|REVEAL|KEY_TRANSFER|ROLE_CHANGE|etc
	Detail    string    // JSON extra info
	IP        string
	UserAgent string
	CreatedAt time.Time
}

// ── DB helpers ────────────────────────────────────────────────

func NewDB(dsn string) (*gorm.DB, error) {
	return gorm.Open(postgres.Open(dsn), &gorm.Config{})
}

func Migrate(db *gorm.DB) error {
	return db.AutoMigrate(
		&User{}, &Session{}, &StegoJob{},
		&KeyRecord{}, &AuditEntry{},
	)
}

// ── Audit logger ──────────────────────────────────────────────

type AuditLogger struct{ db *gorm.DB }

func NewAuditLogger(db *gorm.DB) *AuditLogger { return &AuditLogger{db: db} }

func (a *AuditLogger) Log(ctx context.Context, userID uint, action, detail, ip, ua string) {
	entry := AuditEntry{
		UserID: userID, Action: action, Detail: detail,
		IP: ip, UserAgent: ua, CreatedAt: time.Now(),
	}
	if err := a.db.Create(&entry).Error; err != nil {
		log.Printf("audit log write failed: %v", err)
	}
}