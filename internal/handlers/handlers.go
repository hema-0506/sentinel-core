package handlers

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"
	"time"

	"defense-app/internal/auth"
	"defense-app/internal/models"
	"defense-app/internal/vault"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/jwtauth/v5"
	"github.com/google/uuid"
	"gorm.io/gorm"
)

const maxUploadSize = 1024 << 20 // 1 GB

type Handler struct {
	db          *gorm.DB
	authSvc     *auth.Service
	vault       *vault.KeyVault
	audit       *models.AuditLogger
	tokenAuth   *jwtauth.JWTAuth
	pythonURL   string
	outputDir   string
	pyOutputDir string
}

func New(db *gorm.DB, a *auth.Service, kv *vault.KeyVault,
	al *models.AuditLogger, ta *jwtauth.JWTAuth, py string) *Handler {
	outDir := os.Getenv("OUTPUT_DIR")
	if outDir == "" {
		outDir = "/app/outputs"
	}
	pyOutDir := os.Getenv("PY_OUTPUT_DIR")
	if pyOutDir == "" {
		pyOutDir = "/tmp/defense_outputs"
	}
	return &Handler{
		db: db, authSvc: a, vault: kv, audit: al, tokenAuth: ta,
		pythonURL:   py,
		outputDir:   outDir,
		pyOutputDir: pyOutDir,
	}
}

func (h *Handler) translatePath(pyPath string) string {
	if pyPath == "" {
		return ""
	}
	if len(pyPath) >= len(h.pyOutputDir) && pyPath[:len(h.pyOutputDir)] == h.pyOutputDir {
		return h.outputDir + pyPath[len(h.pyOutputDir):]
	}
	return pyPath
}

// ── Helpers ───────────────────────────────────────────────────

func (h *Handler) json(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func (h *Handler) err(w http.ResponseWriter, status int, msg string) {
	h.json(w, status, map[string]string{"error": msg})
}

func currentUser(r *http.Request) *models.User {
	u, _ := r.Context().Value(models.UserCtxKey).(*models.User)
	return u
}

func ip(r *http.Request) string { return r.RemoteAddr }

func readFileAsBase64(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	mime := "image/png"
	if len(data) > 2 && data[0] == 0xFF && data[1] == 0xD8 {
		mime = "image/jpeg"
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data)
}

func clampInt(n, max int) int {
	if n < 0 {
		return 0
	}
	if n > max {
		return max
	}
	return n
}

func hashBytes(b []byte) string {
	sum := sha256.Sum256(b)
	return fmt.Sprintf("%x", sum)
}

type vaultPayload struct {
	RList      json.RawMessage `json:"r_list"`
	NumSecrets int             `json:"num_secrets"`
	JobID      string          `json:"job_id"`
	StegoHash  string          `json:"stego_hash"`
	StegoB64   string          `json:"stego_b64,omitempty"`
}

// ── Health ────────────────────────────────────────────────────

func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	h.json(w, 200, map[string]string{"status": "ok", "time": time.Now().UTC().Format(time.RFC3339)})
}

// ── Auth ──────────────────────────────────────────────────────

func (h *Handler) registerErr(w http.ResponseWriter, err error) {
	var ve *auth.ValidationError
	if errors.As(err, &ve) {
		h.err(w, 400, ve.Msg)
		return
	}
	h.err(w, 409, err.Error())
}

func (h *Handler) Register(w http.ResponseWriter, r *http.Request) {
	var inp auth.RegisterInput
	if err := json.NewDecoder(r.Body).Decode(&inp); err != nil {
		h.err(w, 400, "invalid JSON")
		return
	}
	u, err := h.authSvc.Register(inp)
	if err != nil {
		h.registerErr(w, err)
		return
	}
	h.audit.Log(r.Context(), u.ID, "REGISTER", "", ip(r), r.UserAgent())
	h.json(w, 201, u)
}

func (h *Handler) Login(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		h.err(w, 400, "invalid JSON")
		return
	}
	res, err := h.authSvc.Login(body.Username, body.Password, ip(r), r.UserAgent())
	if err != nil {
		h.err(w, 401, err.Error())
		return
	}
	h.audit.Log(r.Context(), res.User.ID, "LOGIN", "", ip(r), r.UserAgent())
	h.json(w, 200, map[string]any{
		"user":          res.User,
		"needs_totp":    res.NeedsTOTP,
		"pending_token": res.PendingToken,
		"token":         res.SessionToken,
	})
}

func (h *Handler) VerifyTOTP(w http.ResponseWriter, r *http.Request) {
	var body struct {
		PendingToken string `json:"pending_token"`
		Code         string `json:"code"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		h.err(w, 400, "invalid JSON")
		return
	}
	token, err := h.authSvc.VerifyTOTP(body.PendingToken, body.Code, ip(r), r.UserAgent())
	if err != nil {
		h.err(w, 401, err.Error())
		return
	}
	h.json(w, 200, map[string]string{"token": token})
}

func (h *Handler) SetupTOTP(w http.ResponseWriter, r *http.Request) {
	u := currentUser(r)
	res, err := h.authSvc.SetupTOTP(u.ID)
	if err != nil {
		h.err(w, 500, err.Error())
		return
	}
	h.json(w, 200, res)
}

func (h *Handler) Logout(w http.ResponseWriter, r *http.Request) {
	_, claims, _ := jwtauth.FromContext(r.Context())
	if sid, ok := claims["session_id"].(string); ok {
		h.authSvc.RevokeSession(sid)
	}
	h.json(w, 200, map[string]string{"status": "logged out"})
}

func (h *Handler) Me(w http.ResponseWriter, r *http.Request) {
	h.json(w, 200, currentUser(r))
}

func (h *Handler) BecomeSender(w http.ResponseWriter, r *http.Request) {
	u := currentUser(r)
	if u.Role != "receiver" {
		h.err(w, 400, "only a receiver account can switch to sender")
		return
	}
	if err := h.db.Model(&models.User{}).Where("id = ?", u.ID).
		Update("role", "sender").Error; err != nil {
		h.err(w, 500, err.Error())
		return
	}
	h.audit.Log(r.Context(), u.ID, "ROLE_SELF_UPGRADE",
		`{"from":"receiver","to":"sender"}`, ip(r), r.UserAgent())
	u.Role = "sender"
	h.json(w, 200, u)
}

func (h *Handler) ListSessions(w http.ResponseWriter, r *http.Request) {
	u := currentUser(r)
	var sessions []models.Session
	h.db.Where("user_id = ? AND expires_at > ?", u.ID, time.Now()).Find(&sessions)
	h.json(w, 200, sessions)
}

func (h *Handler) RevokeSession(w http.ResponseWriter, r *http.Request) {
	sid := chi.URLParam(r, "id")
	h.authSvc.RevokeSession(sid)
	h.json(w, 200, map[string]string{"status": "revoked"})
}

// ── Conceal (Sender) ──────────────────────────────────────────

func (h *Handler) Conceal(w http.ResponseWriter, r *http.Request) {
	u := currentUser(r)
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadSize)
	if err := r.ParseMultipartForm(maxUploadSize); err != nil {
		h.err(w, 400, "upload too large")
		return
	}

	jobID := uuid.New().String()
	job := models.StegoJob{ID: jobID, OwnerID: u.ID, Operation: "conceal", Status: "pending"}
	h.db.Create(&job)

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)

	if err := forwardFile(r, mw, "cover"); err != nil {
		h.err(w, 400, "cover image required")
		return
	}

	secretCount := 0
	for i := 0; i < 5; i++ {
		key := fmt.Sprintf("secret_%d", i)
		if _, _, err := r.FormFile(key); err == nil {
			forwardFile(r, mw, key)
			secretCount++
		}
	}
	if secretCount < 2 {
		h.err(w, 400, "at least 2 secret images required")
		return
	}
	mw.Close()

	resp, err := http.Post(h.pythonURL+"/conceal", mw.FormDataContentType(), &buf)
	if err != nil {
		h.db.Model(&job).Updates(map[string]any{"status": "failed", "error_msg": err.Error()})
		h.err(w, 502, "inference service unavailable")
		return
	}
	defer resp.Body.Close()

	var pyRes struct {
		StegoPath    string          `json:"stego_path"`
		StegoB64     string          `json:"stego_b64"`
		RList        json.RawMessage `json:"r_list"`
		RListPath    string          `json:"r_list_path"`
		TextureScore float64         `json:"texture_score"`
		NumSecrets   int             `json:"num_secrets"`
		StegoHash    string          `json:"stego_hash"`
		Error        string          `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&pyRes); err != nil {
		h.err(w, 500, "failed to decode Python response: "+err.Error())
		return
	}

	if pyRes.Error != "" {
		h.db.Model(&job).Updates(map[string]any{"status": "failed", "error_msg": pyRes.Error})
		h.err(w, 422, pyRes.Error)
		return
	}

	// Read key data (prioritize direct HTTP payload over shared volume)
	var rListBytes []byte
	if len(pyRes.RList) > 0 {
		rListBytes = pyRes.RList
	} else {
		localRListPath := h.translatePath(pyRes.RListPath)
		var readErr error
		rListBytes, readErr = os.ReadFile(localRListPath)
		if readErr != nil {
			h.err(w, 500, "failed to read key data: "+readErr.Error())
			return
		}
		defer os.Remove(localRListPath)
	}

	// Obtain stego base64 and hash
	var stegoHash string
	var stegoB64 string
	if pyRes.StegoB64 != "" {
		stegoB64 = pyRes.StegoB64
		stegoHash = pyRes.StegoHash
	} else {
		stegoLocalPath := h.translatePath(pyRes.StegoPath)
		stegoData, readErr := os.ReadFile(stegoLocalPath)
		if readErr != nil {
			h.err(w, 500, "failed to read stego file: "+readErr.Error())
			return
		}
		stegoHash = hashBytes(stegoData)
		stegoB64 = "data:image/png;base64," + base64.StdEncoding.EncodeToString(stegoData)
	}

	payload, err := json.Marshal(vaultPayload{
		RList:      json.RawMessage(rListBytes),
		NumSecrets: secretCount,
		JobID:      jobID,
		StegoHash:  stegoHash,
		StegoB64:   stegoB64,
	})
	if err != nil {
		h.err(w, 500, "failed to encode vault payload")
		return
	}

	keyID, err := h.vault.Store(jobID, u.ID, string(payload))
	if err != nil {
		h.err(w, 500, "key storage failed")
		return
	}

	h.db.Model(&job).Updates(map[string]any{
		"status":        "done",
		"num_secrets":   secretCount,
		"stego_path":    pyRes.StegoPath,
		"texture_score": pyRes.TextureScore,
	})

	h.audit.Log(r.Context(), u.ID, "CONCEAL",
		fmt.Sprintf(`{"job_id":"%s","secrets":%d}`, jobID, secretCount),
		ip(r), r.UserAgent())

	h.json(w, 200, map[string]any{
		"job_id":          jobID,
		"key_id":          keyID,
		"texture_score":   pyRes.TextureScore,
		"num_secrets":     secretCount,
		"stego_image_b64": stegoB64,
		"message":         "Embedding complete. Share key_id securely with receiver.",
	})
}

// ── Reveal (Receiver) ─────────────────────────────────────────

func (h *Handler) Reveal(w http.ResponseWriter, r *http.Request) {
	u := currentUser(r)
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadSize)
	if err := r.ParseMultipartForm(maxUploadSize); err != nil {
		h.err(w, 400, "upload too large")
		return
	}
	keyID := r.FormValue("key_id")

	vaultBytes, err := h.vault.Retrieve(keyID, u.ID)
	if err != nil {
		h.err(w, 403, err.Error())
		return
	}

	var payload vaultPayload
	if jsonErr := json.Unmarshal(vaultBytes, &payload); jsonErr != nil {
		payload.RList = json.RawMessage(vaultBytes)
		payload.NumSecrets = 0
		payload.JobID = ""
		payload.StegoHash = ""
	}
	numSecrets := payload.NumSecrets

	stegoFileHandle, stegoHeader, err := r.FormFile("stego")
	if err != nil {
		h.err(w, 400, "stego image required")
		return
	}
	stegoBytes, err := io.ReadAll(stegoFileHandle)
	stegoFileHandle.Close()
	if err != nil {
		h.err(w, 400, "failed to read stego image")
		return
	}

	uploadedHash := hashBytes(stegoBytes)
	verified := false
	canVerify := false

	if payload.StegoHash != "" {
		canVerify = true
		if uploadedHash == payload.StegoHash {
			verified = true
		}
	}

	if !verified && payload.JobID != "" {
		var concealJob models.StegoJob
		if dbErr := h.db.Where("id = ? AND operation = 'conceal'", payload.JobID).
			First(&concealJob).Error; dbErr == nil && concealJob.StegoPath != "" {
			storedPath := h.translatePath(concealJob.StegoPath)
			if storedData, readErr := os.ReadFile(storedPath); readErr == nil {
				canVerify = true
				if uploadedHash == hashBytes(storedData) {
					verified = true
				}
			}
		}
	}

	if canVerify && !verified {
		h.audit.Log(r.Context(), u.ID, "REVEAL_REJECTED",
			fmt.Sprintf(`{"key_id":"%s","reason":"stego_mismatch"}`, keyID),
			ip(r), r.UserAgent())
		h.err(w, 403, "stego image does not match this key — wrong image uploaded")
		return
	}

	jobID := uuid.New().String()
	job := models.StegoJob{ID: jobID, OwnerID: u.ID, Operation: "reveal", Status: "pending"}
	h.db.Create(&job)

	// Build multipart body for Python — send stego bytes and raw r_list over HTTP
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)

	fw, err := mw.CreateFormFile("stego", stegoHeader.Filename)
	if err != nil {
		h.err(w, 500, "multipart error")
		return
	}
	fw.Write(stegoBytes)

	// Stream r_list as a form file to bypass Werkzeug 500KB text field cap
	rfw, err := mw.CreateFormFile("r_list_file", "r_list.json")
	if err != nil {
		h.err(w, 500, "multipart error")
		return
	}
	rfw.Write([]byte(payload.RList))

	mw.Close()

	resp, err := http.Post(h.pythonURL+"/reveal", mw.FormDataContentType(), &buf)
	if err != nil {
		h.db.Model(&job).Updates(map[string]any{"status": "failed"})
		h.err(w, 502, "inference service unavailable")
		return
	}
	defer resp.Body.Close()

	bodyBytes, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		h.db.Model(&job).Updates(map[string]any{"status": "failed", "error_msg": string(bodyBytes)})
		h.err(w, resp.StatusCode, "Python Error: "+string(bodyBytes))
		return
	}

	var pyRes struct {
		CoverPath    string    `json:"cover_path"`
		CoverB64     string    `json:"cover_b64"`
		StegoPSNR    float64   `json:"stego_psnr"`
		StegoMSE     float64   `json:"stego_mse"`
		SecretsPaths []string  `json:"secrets_paths"`
		SecretsB64   []string  `json:"secrets_b64"`
		PSNRScores   []float64 `json:"psnr_scores"`
		SSIMScores   []float64 `json:"ssim_scores"`
		Error        string    `json:"error"`
	}
	json.Unmarshal(bodyBytes, &pyRes)

	if pyRes.Error != "" {
		h.db.Model(&job).Updates(map[string]any{"status": "failed", "error_msg": pyRes.Error})
		h.err(w, 422, pyRes.Error)
		return
	}

	h.db.Model(&job).Updates(map[string]any{"status": "done"})

	h.audit.Log(r.Context(), u.ID, "REVEAL",
		fmt.Sprintf(`{"job_id":"%s","key_id":"%s"}`, jobID, keyID),
		ip(r), r.UserAgent())

	totalFromPython := len(pyRes.SecretsPaths)
	if totalFromPython == 0 && len(pyRes.SecretsB64) > 0 {
		totalFromPython = len(pyRes.SecretsB64)
	}

	if numSecrets <= 0 {
		var keyRec models.KeyRecord
		if dbErr := h.db.Where("id = ?", keyID).First(&keyRec).Error; dbErr == nil {
			var concealJob models.StegoJob
			if dbErr2 := h.db.Where("id = ? AND operation = 'conceal'", keyRec.JobID).
				First(&concealJob).Error; dbErr2 == nil && concealJob.NumSecrets > 0 {
				numSecrets = concealJob.NumSecrets
			}
		}
		if numSecrets <= 0 {
			numSecrets = totalFromPython
		}
	}

	n := clampInt(numSecrets, totalFromPython)

	psnrScores := pyRes.PSNRScores
	if len(psnrScores) > n {
		psnrScores = psnrScores[:n]
	}
	ssimScores := pyRes.SSIMScores
	if len(ssimScores) > n {
		ssimScores = ssimScores[:n]
	}

	// Read cover image (prefer HTTP base64 over file read)
	coverB64 := pyRes.CoverB64
	if coverB64 == "" {
		coverB64 = readFileAsBase64(h.translatePath(pyRes.CoverPath))
	}

	// Read recovered secrets (prefer HTTP base64 over file read)
	secretsB64 := make([]string, n)
	if len(pyRes.SecretsB64) >= n {
		copy(secretsB64, pyRes.SecretsB64[:n])
	} else {
		secretPaths := pyRes.SecretsPaths[:n]
		for i, path := range secretPaths {
			secretsB64[i] = readFileAsBase64(h.translatePath(path))
		}
	}

	h.json(w, 200, map[string]any{
		"job_id":      jobID,
		"cover_b64":   coverB64,
		"stego_psnr":  pyRes.StegoPSNR,
		"stego_mse":   pyRes.StegoMSE,
		"psnr_scores": psnrScores,
		"ssim_scores": ssimScores,
		"secrets_b64": secretsB64,
		"num_secrets": n,
		"message":     "Reveal complete.",
	})
}

// ── Key transfer ──────────────────────────────────────────────

func (h *Handler) TransferKey(w http.ResponseWriter, r *http.Request) {
	u := currentUser(r)
	jobID := chi.URLParam(r, "job_id")
	var body struct {
		ReceiverUsername string `json:"receiver_username"`
	}
	json.NewDecoder(r.Body).Decode(&body)

	var receiver models.User
	if err := h.db.Where("username = ? AND role IN ('receiver','admin')", body.ReceiverUsername).
		First(&receiver).Error; err != nil {
		h.err(w, 404, "receiver not found")
		return
	}

	var key models.KeyRecord
	if err := h.db.Where("job_id = ? AND owner_id = ?", jobID, u.ID).First(&key).Error; err != nil {
		h.err(w, 404, "key not found")
		return
	}

	if err := h.vault.Transfer(key.ID, u.ID, receiver.ID); err != nil {
		h.err(w, 500, err.Error())
		return
	}

	h.audit.Log(r.Context(), u.ID, "KEY_TRANSFER",
		fmt.Sprintf(`{"key_id":"%s","to_user":%d}`, key.ID, receiver.ID),
		ip(r), r.UserAgent())

	h.json(w, 200, map[string]any{
		"key_id":  key.ID,
		"message": fmt.Sprintf("Key transferred to %s", receiver.Username),
	})
}

func (h *Handler) KeyInbox(w http.ResponseWriter, r *http.Request) {
	u := currentUser(r)
	var keys []models.KeyRecord
	h.db.Where("transferred_to = ?", u.ID).Find(&keys)
	h.json(w, 200, keys)
}

// ── Jobs ──────────────────────────────────────────────────────

func (h *Handler) ListJobs(w http.ResponseWriter, r *http.Request) {
	u := currentUser(r)
	var jobs []models.StegoJob
	h.db.Where("owner_id = ?", u.ID).Order("created_at desc").Limit(50).Find(&jobs)
	h.json(w, 200, jobs)
}

func (h *Handler) GetJob(w http.ResponseWriter, r *http.Request) {
	u := currentUser(r)
	id := chi.URLParam(r, "id")
	var job models.StegoJob
	if err := h.db.Where("id = ? AND owner_id = ?", id, u.ID).First(&job).Error; err != nil {
		h.err(w, 404, "not found")
		return
	}
	h.json(w, 200, job)
}

func (h *Handler) DownloadStego(w http.ResponseWriter, r *http.Request) {
	u := currentUser(r)
	id := chi.URLParam(r, "id")
	var job models.StegoJob
	if err := h.db.Where("id = ? AND owner_id = ?", id, u.ID).First(&job).Error; err != nil {
		h.err(w, 404, "job not found")
		return
	}
	if job.StegoPath == "" {
		h.err(w, 404, "stego file not available")
		return
	}
	data, err := os.ReadFile(job.StegoPath)
	if err != nil {
		h.err(w, 404, "stego file not found on disk")
		return
	}
	h.audit.Log(r.Context(), u.ID, "DOWNLOAD_STEGO",
		fmt.Sprintf(`{"job_id":"%s"}`, id), ip(r), r.UserAgent())
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="stego_%s.png"`, id[:8]))
	w.WriteHeader(200)
	w.Write(data)
}

func (h *Handler) DownloadSecrets(w http.ResponseWriter, r *http.Request) {
	h.err(w, 501, "individual secret download: use the base64 images returned by /reveal")
}

// ── Admin ─────────────────────────────────────────────────────

func (h *Handler) ListUsers(w http.ResponseWriter, r *http.Request) {
	var users []models.User
	h.db.Find(&users)
	h.json(w, 200, users)
}

func (h *Handler) CreateUser(w http.ResponseWriter, r *http.Request) {
	var inp auth.RegisterInput
	json.NewDecoder(r.Body).Decode(&inp)
	u, err := h.authSvc.Register(inp)
	if err != nil {
		h.registerErr(w, err)
		return
	}
	h.json(w, 201, u)
}

func (h *Handler) UpdateUserRole(w http.ResponseWriter, r *http.Request) {
	admin := currentUser(r)
	uid, _ := strconv.Atoi(chi.URLParam(r, "id"))
	var body struct {
		Role string `json:"role"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	if err := h.db.Model(&models.User{}).Where("id = ?", uid).
		Update("role", body.Role).Error; err != nil {
		h.err(w, 500, err.Error())
		return
	}
	h.audit.Log(r.Context(), admin.ID, "ROLE_CHANGE",
		fmt.Sprintf(`{"target_user":%d,"new_role":"%s"}`, uid, body.Role),
		ip(r), r.UserAgent())
	h.json(w, 200, map[string]string{"status": "updated"})
}

func (h *Handler) DeleteUser(w http.ResponseWriter, r *http.Request) {
	uid, _ := strconv.Atoi(chi.URLParam(r, "id"))
	h.db.Model(&models.User{}).Where("id = ?", uid).Update("active", false)
	h.json(w, 200, map[string]string{"status": "deactivated"})
}

func (h *Handler) AuditLog(w http.ResponseWriter, r *http.Request) {
	page, _ := strconv.Atoi(r.URL.Query().Get("page"))
	if page < 1 {
		page = 1
	}
	var entries []models.AuditEntry
	h.db.Order("created_at desc").Offset((page-1)*100).Limit(100).Find(&entries)
	h.json(w, 200, entries)
}

func (h *Handler) RotateMasterKey(w http.ResponseWriter, r *http.Request) {
	h.err(w, 501, "key rotation requires ops procedure")
}

// ── Util ──────────────────────────────────────────────────────

func forwardFile(r *http.Request, mw *multipart.Writer, fieldName string) error {
	file, header, err := r.FormFile(fieldName)
	if err != nil {
		return err
	}
	defer file.Close()
	fw, err := mw.CreateFormFile(fieldName, header.Filename)
	if err != nil {
		return err
	}
	_, err = io.Copy(fw, file)
	return err
}