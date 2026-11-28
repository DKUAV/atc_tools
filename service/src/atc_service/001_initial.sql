CREATE TABLE accounts (
    username text PRIMARY KEY
);

CREATE TABLE tokens (
    digest bytea PRIMARY KEY,
    username text NOT NULL REFERENCES accounts(username)
);

CREATE TABLE assets (
    id uuid PRIMARY KEY,
    owner text NOT NULL REFERENCES accounts(username),
    kind text NOT NULL CHECK (kind IN ('onnx', 'inspection', 'om', 'simulated_om', 'build_info')),
    filename text NOT NULL,
    size bigint NOT NULL CHECK (size >= 0),
    sha256 text NOT NULL,
    validated boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE jobs (
    id uuid PRIMARY KEY,
    owner text NOT NULL REFERENCES accounts(username),
    operation text NOT NULL CHECK (operation IN ('inspect_onnx', 'compile_om')),
    input_id uuid NOT NULL REFERENCES assets(id),
    image text NOT NULL,
    preset text,
    idempotency_key text,
    simulated boolean NOT NULL DEFAULT false,
    state text NOT NULL DEFAULT 'queued' CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    cancel_requested boolean NOT NULL DEFAULT false,
    error text,
    exit_code integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    UNIQUE (owner, idempotency_key)
);

CREATE TABLE attempts (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES jobs(id),
    started_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE jobs ADD COLUMN attempt_id uuid REFERENCES attempts(id);

CREATE TABLE job_outputs (
    job_id uuid NOT NULL REFERENCES jobs(id),
    asset_id uuid NOT NULL REFERENCES assets(id),
    PRIMARY KEY (job_id, asset_id)
);

CREATE INDEX queued_jobs ON jobs (created_at, id) WHERE state = 'queued';
CREATE INDEX owner_jobs ON jobs (owner, created_at DESC);
CREATE INDEX job_attempts ON attempts (job_id, started_at DESC);
