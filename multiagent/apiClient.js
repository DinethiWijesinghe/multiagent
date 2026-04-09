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
  if (hasBody && !headers['Content-Type']) {
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
    const timerId = timeoutMs > 0 ? setTimeout(() => perCtrl.abort(), timeoutMs) : null;
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
      lastError = error;
    } finally {
      if (timerId) clearTimeout(timerId);
      outerSignal?.removeEventListener('abort', onOuterAbort);
    }
  }

  if (lastResponse) return lastResponse;
  const targets = API_BASES.map((v) => v || '<same-origin>').join(', ');
  throw lastError || new Error(`Cannot reach API server (${targets}). Start the backend locally on port 8000 or update VITE_API_URL.`);
}

export async function apiErrorMessage(response, fallbackMessage) {
  const fallback = fallbackMessage || `Error ${response?.status ?? ''}`.trim();
  try {
    const payload = await response.json();
    return payload?.detail || payload?.message || fallback;
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
