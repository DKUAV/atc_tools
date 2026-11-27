ALTER TABLE accounts ADD COLUMN password_hash bytea;

CREATE TABLE browser_sessions (
    id uuid PRIMARY KEY,
    owner text NOT NULL REFERENCES accounts(username),
    token_digest bytea NOT NULL UNIQUE,
    expires_at timestamptz NOT NULL,
    closed boolean NOT NULL DEFAULT false
);

CREATE TABLE browser_login_attempts (
    owner text PRIMARY KEY REFERENCES accounts(username),
    attempts integer NOT NULL DEFAULT 1,
    reset_at timestamptz NOT NULL DEFAULT (now() + interval '1 minute')
);

ALTER TABLE assets ADD COLUMN browser_session_id uuid REFERENCES browser_sessions(id);
ALTER TABLE assets ADD COLUMN cleared boolean NOT NULL DEFAULT false;
ALTER TABLE jobs ADD COLUMN browser_session_id uuid REFERENCES browser_sessions(id);

CREATE INDEX live_session_assets ON assets (browser_session_id)
    WHERE browser_session_id IS NOT NULL AND NOT cleared;
CREATE INDEX active_session_jobs ON jobs (browser_session_id)
    WHERE state IN ('queued', 'running');
