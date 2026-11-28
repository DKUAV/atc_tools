CREATE TABLE accounts (
    username text PRIMARY KEY,
    password_hash blob
);

CREATE TABLE tokens (
    digest blob PRIMARY KEY,
    username text NOT NULL REFERENCES accounts(username)
);

CREATE TABLE browser_sessions (
    id uuid PRIMARY KEY,
    owner text NOT NULL REFERENCES accounts(username),
    token_digest blob NOT NULL UNIQUE,
    expires_at text NOT NULL,
    closed boolean NOT NULL DEFAULT false
);

CREATE TABLE browser_login_attempts (
    owner text PRIMARY KEY REFERENCES accounts(username),
    attempts integer NOT NULL DEFAULT 1,
    reset_at text NOT NULL DEFAULT (datetime('now','+1 minute'))
);

CREATE TABLE assets (
    id uuid PRIMARY KEY,
    owner text NOT NULL REFERENCES accounts(username),
    kind text NOT NULL CHECK (kind IN ('onnx','inspection','om','simulated_om','build_info','dataset','accuracy','profiling','quantized_onnx','tensors','baseline_om')),
    filename text NOT NULL,
    size integer NOT NULL CHECK (size >= 0),
    sha256 text NOT NULL,
    browser_session_id uuid REFERENCES browser_sessions(id),
    cleared boolean NOT NULL DEFAULT false,
    validated boolean NOT NULL DEFAULT false,
    created_at text NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE jobs (
    id uuid PRIMARY KEY,
    owner text NOT NULL REFERENCES accounts(username),
    operation text NOT NULL CHECK (operation IN ('inspect_onnx','compile_om','quantize_onnx','run_cpu','run_npu','compare_outputs','profile_om','retry_step')),
    input_id uuid NOT NULL REFERENCES assets(id),
    image text NOT NULL,
    preset text,
    idempotency_key text,
    simulated boolean NOT NULL DEFAULT false,
    state text NOT NULL DEFAULT 'queued' CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    cancel_requested boolean NOT NULL DEFAULT false,
    error text,
    exit_code integer,
    created_at text NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    started_at text,
    finished_at text,
    workflow json,
    stages json NOT NULL DEFAULT '{}',
    stage_times json NOT NULL DEFAULT '{}',
    accuracy_summary json,
    browser_session_id uuid REFERENCES browser_sessions(id),
    attempt_id uuid REFERENCES attempts(id),
    lane text NOT NULL CHECK (lane IN ('cpu','compile','device')),
    UNIQUE (owner, idempotency_key)
);

CREATE TABLE attempts (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES jobs(id),
    started_at text NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE job_outputs (
    job_id uuid NOT NULL REFERENCES jobs(id),
    asset_id uuid NOT NULL REFERENCES assets(id),
    PRIMARY KEY (job_id, asset_id)
);

CREATE INDEX queued_jobs ON jobs (created_at, id) WHERE state = 'queued';
CREATE INDEX owner_jobs ON jobs (owner, created_at DESC);
CREATE INDEX job_attempts ON attempts (job_id, started_at DESC);
CREATE INDEX live_session_assets ON assets (browser_session_id)
    WHERE browser_session_id IS NOT NULL AND NOT cleared;
CREATE INDEX active_session_jobs ON jobs (browser_session_id)
    WHERE state IN ('queued', 'running');

CREATE INDEX queued_jobs_by_lane ON jobs (lane, created_at, id) WHERE state = 'queued';

CREATE TABLE worker_status (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    heartbeat_at text NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    checks json NOT NULL
);
