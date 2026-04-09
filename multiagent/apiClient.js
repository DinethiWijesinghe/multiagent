const CONFIGURED_API_URL = (import.meta.env.VITE_API_URL || '').trim().replace(/\/$/, '');
const DEFAULT_LOCAL_API_URL = 'http://127.0.0.1:8000';
const SAME_ORIGIN_API_URL = typeof window !== 'undefined'
  ? `${window.location.protocol}//${window.location.host}`.replace(/\/$/, '')
  : '';

const API_BASES = CONFIGURED_API_URL
  ? [CONFIGURED_API_URL]
  : Array.from(
      new Set(
        [
          DEFAULT_LOCAL_API_URL,
          SAME_ORIGIN_API_URL,
        ].filter((v) => v !== undefined && v !== null && v !== '')
      )
    );

const PER_URL_TIMEOUT_MS = 20000;

function isMultipartLikeBody(body) {
  return typeof FormData !== 'undefined' && body instanceof FormData;
}

function formatApiDetail(detail, fallbackMessage) {
  if (typeof detail === 'string' && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length) {
    const messages = detail
      .map((item) => {
        if (typeof item === 'string' && item.trim()) {
          return item;
        }
        if (item && typeof item === 'object') {
          const loc = Array.isArray(item.loc) ? item.loc.slice(1).join('.') : '';
          const msg = typeof item.msg === 'string' ? item.msg : JSON.stringify(item);
          return loc ? `${loc}: ${msg}` : msg;
        }
        return '';
      })
      .filter(Boolean);
    if (messages.length) {
      return messages.join('; ');
    }
  }
  if (detail && typeof detail === 'object') {
    try {
      return JSON.stringify(detail);
    } catch {
      return fallbackMessage;
    }
  }
  return fallbackMessage;
}

export async function apiFetch(path, tokenOrOptions, maybeOptions) {
  const token = typeof tokenOrOptions === 'string' || tokenOrOptions == null
    ? tokenOrOptions
    : undefined;
  const options = token === undefined
    ? (tokenOrOptions || {})
    : (maybeOptions || {});

  const timeoutMs = Number.isFinite(options?.timeoutMs) ? options.timeoutMs : PER_URL_TIMEOUT_MS;
  const fetchOptions = { ...(options || {}) };
  delete fetchOptions.timeoutMs;

  const headers = { ...(fetchOptions.headers || {}) };
  const hasBody = fetchOptions.body !== undefined && fetchOptions.body !== null;
  if (hasBody && !headers['Content-Type'] && !isMultipartLikeBody(fetchOptions.body)) {
    headers['Content-Type'] = 'application/json';
  }
  if (token && !headers.Authorization) {
    headers.Authorization = `Bearer ${token}`;
  }

  let lastError;
  let lastResponse;
  const outerSignal = fetchOptions?.signal;
  for (const baseUrl of API_BASES) {
    if (outerSignal?.aborted) break;
    const perCtrl = new AbortController();
    const target = baseUrl ? `${baseUrl}${path}` : path;
    let timedOut = false;
    const timerId = timeoutMs > 0 ? setTimeout(() => {
      timedOut = true;
      perCtrl.abort();
    }, timeoutMs) : null;
    const onOuterAbort = () => perCtrl.abort();
    outerSignal?.addEventListener('abort', onOuterAbort);

    try {
      const response = await fetch(target, { ...fetchOptions, headers, signal: perCtrl.signal });
      const contentType = (response.headers.get('content-type') || '').toLowerCase();
      const isHtmlResponse = contentType.includes('text/html');
      if ((response.status === 404 || isHtmlResponse) && API_BASES.length > 1) {
        lastResponse = response;
        continue;
      }
      return response;
    } catch (error) {
      const message = String(error?.message || error || '').toLowerCase();
      const aborted = error?.name === 'AbortError' || message.includes('aborted');
      if (aborted && timedOut) {
        const sec = Math.max(1, Math.round(timeoutMs / 1000));
        lastError = new Error(`Request timed out after ${sec}s for ${path}`);
      } else if (aborted) {
        lastError = new Error(`Request was cancelled for ${path}`);
      } else {
        lastError = error;
      }
    } finally {
      if (timerId) clearTimeout(timerId);
      outerSignal?.removeEventListener('abort', onOuterAbort);
    }
  }

  if (lastResponse) {
    const ct = (lastResponse.headers.get('content-type') || '').toLowerCase();
    if (!ct.includes('text/html')) return lastResponse; // e.g. 404 with JSON error body
    // All bases returned HTML (SPA fallback) — backend is unreachable
    const targets = API_BASES.map((v) => v || '<same-origin>').join(', ');
    throw lastError || new Error(`Cannot reach API server (${targets}). Start the backend on port 8000 or update VITE_API_URL.`);
  }
  const targets = API_BASES.map((v) => v || '<same-origin>').join(', ');
  throw lastError || new Error(`Cannot reach API server (${targets}). Start the backend locally on port 8000 or update VITE_API_URL.`);
}

export async function apiErrorMessage(response, fallbackMessage) {
  const fallback = fallbackMessage || `Error ${response?.status ?? ''}`.trim();
  try {
    const payload = await response.json();
    return formatApiDetail(payload?.detail, payload?.message || fallback);
  } catch {
    return fallback;
  }
}

export async function apiJson(path, tokenOrOptions, maybeOptions) {
  const response = await apiFetch(path, tokenOrOptions, maybeOptions);
  if (!response.ok) {
    throw new Error(await apiErrorMessage(response));
  }
  return response.json();
}
