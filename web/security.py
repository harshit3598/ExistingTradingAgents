"""Request hardening for the web UI.

The app can store API keys, read the results tree and — with live trading
enabled — place real orders, so every request is checked before it reaches a
route:

- **Host allow-list** (DNS-rebinding guard): unless token auth is on, only
  loopback names (plus ``TRADINGAGENTS_WEB_ALLOWED_HOSTS``) may address the
  server, so a hostile page that rebinds its own domain to 127.0.0.1 is refused.
- **CSRF guard**: state-changing requests must be ``application/json`` (which a
  cross-site page cannot send without a CORS preflight this app never grants)
  and, when the browser sends an ``Origin``, it must match the request host.
- **Optional token auth** (``TRADINGAGENTS_WEB_TOKEN``): required for anything
  but a loopback bind. Opening ``/?token=…`` once stores it in an HttpOnly,
  SameSite=Strict cookie; API clients may send ``Authorization: Bearer …``.
- **Security headers**: a strict CSP (no inline script, no framing — the live
  order button must never be clickjackable), nosniff and no referrer.
"""

from __future__ import annotations

import hmac
import os
import secrets
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

TOKEN_ENV = "TRADINGAGENTS_WEB_TOKEN"
ALLOWED_HOSTS_ENV = "TRADINGAGENTS_WEB_ALLOWED_HOSTS"
COOKIE_NAME = "ta_session"

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def is_loopback(host: str) -> bool:
    return host.strip("[]").lower() in LOOPBACK_HOSTS


def ensure_token(bind_host: str) -> str | None:
    """The access token for this server, minting one for non-loopback binds.

    A loopback-only server is reachable by local processes alone and keeps the
    zero-setup flow; anything wider (``0.0.0.0``, a LAN address, Docker) must
    authenticate, so a token is generated when none is configured.
    """
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token and not is_loopback(bind_host):
        token = secrets.token_urlsafe(24)
        os.environ[TOKEN_ENV] = token
    return token or None


def _hostname(value: str) -> str:
    """``host[:port]`` → lower-case host name (IPv6 brackets stripped)."""
    try:
        return (urlsplit(f"//{value}").hostname or "").lower()
    except ValueError:
        return ""


def _allowed_hosts() -> set[str]:
    extra = os.environ.get(ALLOWED_HOSTS_ENV, "")
    return set(LOOPBACK_HOSTS) | {h.strip().lower() for h in extra.split(",") if h.strip()}


def _error(status: int, detail: str) -> Response:
    return JSONResponse({"detail": detail}, status_code=status)


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = self._check(request)
        if response is None:
            response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    def _check(self, request: Request) -> Response | None:
        token = os.environ.get(TOKEN_ENV, "").strip()
        host_header = request.headers.get("host", "")

        if token:
            supplied = request.query_params.get("token")
            if supplied is not None and request.method == "GET" and request.url.path == "/":
                if not hmac.compare_digest(supplied, token):
                    return _error(401, "Invalid access token.")
                # Swap the URL token for a cookie so it leaves the address bar.
                redirect = RedirectResponse("/", status_code=303)
                redirect.set_cookie(
                    COOKIE_NAME, token, httponly=True, samesite="strict",
                    secure=request.url.scheme == "https", max_age=30 * 86400,
                )
                return redirect
            presented = request.cookies.get(COOKIE_NAME, "")
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                presented = auth[7:].strip()
            if not presented or not hmac.compare_digest(presented, token):
                return _error(
                    401, "Authentication required: open the URL printed at server "
                         "start (…/?token=…) or send Authorization: Bearer <token>."
                )
        elif _hostname(host_header) not in _allowed_hosts():
            return _error(
                400, f"Host not allowed. Add it to {ALLOWED_HOSTS_ENV} (and set "
                     f"{TOKEN_ENV}) to serve beyond localhost."
            )

        if request.method not in SAFE_METHODS:
            # Same-origin means same host *and* port; "null" (sandboxed
            # frames, file://) never matches.
            origin = request.headers.get("origin")
            if origin is not None and urlsplit(origin).netloc.lower() != host_header.lower():
                return _error(403, "Cross-origin request refused.")
            content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type != "application/json":
                return _error(415, "Requests must be sent as application/json.")
        return None
