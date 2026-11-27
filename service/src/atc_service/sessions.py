"""Provision browser passwords and manage revocable, server-side sessions."""

from dataclasses import dataclass
from hashlib import pbkdf2_hmac, sha256
from secrets import compare_digest, token_bytes, token_urlsafe
from uuid import UUID, uuid4

from atc_service.store import ResourceError, Store


@dataclass(frozen=True)
class BrowserSession:
    """Identify the account and file scope authorized by a browser cookie."""

    id: UUID
    owner: str
    csrf: str


def _password_digest(password: str, salt: bytes) -> bytes:
    """Derive the password verifier with a per-account salt.

    Parameters
    ----------
    password
        User-entered password.
    salt
        Sixteen bytes of salt.

    Returns
    -------
    bytes
        Salt followed by the derived digest.
    """
    return salt + pbkdf2_hmac("sha256", password.encode(), salt, 600_000)


def set_browser_password(store: Store, owner: str, password: str) -> None:
    """Set an operator-provisioned password and revoke prior browser sessions.

    Parameters
    ----------
    store
        Deployment database store.
    owner
        Account name to provision or update.
    password
        Password supplied at the operator prompt.
    """
    verifier = _password_digest(password, token_bytes(16))
    with store.connect() as db:
        db.execute(
            "INSERT INTO accounts (username, password_hash) VALUES (%s, %s) "
            "ON CONFLICT (username) DO UPDATE SET password_hash = EXCLUDED.password_hash",
            (owner, verifier),
        )
        db.execute(
            "UPDATE browser_sessions SET closed = true WHERE owner = %s", (owner,)
        )


def open_session(store: Store, owner: str, password: str, previous: str) -> str:
    """Verify a password and issue a new browser session cookie.

    Parameters
    ----------
    store
        Deployment database store.
    owner
        Account name from the sign-in form.
    password
        Password from the sign-in form.
    previous
        Existing cookie to revoke when sign-in succeeds.

    Returns
    -------
    str
        The new opaque session cookie.

    Raises
    ------
    ResourceError
        Credentials do not match or too many attempts were made.
    """
    with store.connect() as db:
        account = db.execute(
            "SELECT password_hash FROM accounts WHERE username = %s", (owner,)
        ).fetchone()
        verifier = account["password_hash"] if account else None
        limited = False
        if verifier is not None:
            attempt = db.execute(
                "INSERT INTO browser_login_attempts (owner) VALUES (%s) "
                "ON CONFLICT (owner) DO UPDATE SET "
                "attempts = CASE WHEN browser_login_attempts.reset_at <= now() THEN 1 "
                "ELSE LEAST(browser_login_attempts.attempts + 1, 9) END, "
                "reset_at = CASE WHEN browser_login_attempts.reset_at <= now() "
                "THEN now() + interval '1 minute' ELSE browser_login_attempts.reset_at END "
                "RETURNING attempts",
                (owner,),
            ).fetchone()
            assert attempt is not None
            limited = attempt["attempts"] > 8
    if limited:
        raise ResourceError(429, "Too many sign-in attempts. Try again in one minute.")
    expected = verifier if verifier is not None else bytes(48)
    if (
        not compare_digest(_password_digest(password, expected[:16]), expected)
        or verifier is None
    ):
        raise ResourceError(401, "Account name or password is incorrect.")
    token = token_urlsafe(32)
    with store.connect() as db:
        current = db.execute(
            "SELECT 1 FROM accounts WHERE username = %s AND password_hash = %s FOR UPDATE",
            (owner, verifier),
        ).fetchone()
        if current is None:
            raise ResourceError(401, "Account name or password is incorrect.")
        db.execute("DELETE FROM browser_login_attempts WHERE owner = %s", (owner,))
        db.execute(
            "INSERT INTO browser_sessions (id, owner, token_digest, expires_at) "
            "VALUES (%s, %s, %s, now() + make_interval(secs => %s))",
            (
                uuid4(),
                owner,
                sha256(token.encode()).digest(),
                store.settings.browser_session_seconds,
            ),
        )
        if len(previous) == 43:
            db.execute(
                "UPDATE browser_sessions SET closed = true WHERE token_digest = %s",
                (sha256(previous.encode()).digest(),),
            )
    return token


def read_session(store: Store, token: str) -> BrowserSession:
    """Resolve a live session and derive its request verification token.

    Parameters
    ----------
    store
        Deployment database store.
    token
        Opaque browser cookie.

    Returns
    -------
    BrowserSession
        Current account, file scope, and CSRF token.

    Raises
    ------
    ResourceError
        No active session matches the cookie.
    """
    if len(token) == 43:
        with store.connect() as db:
            row = db.execute(
                "SELECT id, owner FROM browser_sessions WHERE token_digest = %s "
                "AND NOT closed AND expires_at > now()",
                (sha256(token.encode()).digest(),),
            ).fetchone()
        if row is not None:
            return BrowserSession(
                row["id"], row["owner"], sha256(b"csrf:" + token.encode()).hexdigest()
            )
    raise ResourceError(401, "Sign in to continue.")


def close_session(store: Store, session: BrowserSession) -> None:
    """Revoke a session without deleting account-owned job history.

    Parameters
    ----------
    store
        Deployment database store.
    session
        Session to revoke.
    """
    with store.connect() as db:
        db.execute(
            "UPDATE browser_sessions SET closed = true WHERE id = %s", (session.id,)
        )
