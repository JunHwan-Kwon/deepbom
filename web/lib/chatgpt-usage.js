const CONSENT_VERSION = '2026-09-17';
const LIFETIME = 90 * 86400000;

// No model object, filename, hash, attachment URL, or host user metadata enters
// this module. Callers pass only an allowlisted event, format, and export/view.
export function createChatGptUsage({ endpoint, fetcher = globalThis.fetch.bind(globalThis), storage, now = Date.now, channel = 'chatgpt' } = {}) {
  if (!['chatgpt', 'web'].includes(channel)) throw new Error('Unsupported usage channel');
  const storageKey = `deepbom.${channel}.usage.v1`;
  if (storage === undefined) { try { storage = globalThis.localStorage; } catch { storage = null; } }
  let preference = { enabled: false, cohort: 'public' };
  try {
    const saved = JSON.parse(storage?.getItem(storageKey) || 'null');
    if (saved?.version === CONSENT_VERSION && ['public', 'test'].includes(saved.cohort)) {
      preference.cohort = saved.cohort;
      if (saved.expires > now() && /^[A-Za-z0-9_-]{43}$/.test(saved.key || '')) preference = saved;
    }
  } catch { /* Storage access is optional. */ }
  const makeRun = () => ({ session: null, busy: null, events: new Map(), sent: new Set() });
  let run = makeRun();
  const active = new Set();
  let revision = 0;
  let persistent = false;
  let lastError = '';
  let listener = () => {};
  const remember = () => {
    try { storage?.setItem(storageKey, JSON.stringify(preference)); return Boolean(storage && storage.getItem(storageKey) === JSON.stringify(preference)); }
    catch { return false; }
  };
  if (preference.enabled) persistent = remember();
  const post = async (path, body) => {
    const response = await fetcher(new URL(path, endpoint), {
      method: 'POST', credentials: 'omit', referrerPolicy: 'no-referrer',
      headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
      signal: AbortSignal.timeout(8000),
    });
    if (!response.ok) throw new Error('Usage statistics could not be saved. Analysis is unaffected.');
    return response.json();
  };
  const flush = (target = run) => {
    const { events, sent } = target;
    if (preference.enabled && preference.expires <= now()) {
      preference.enabled = false; revision += 1; remember(); listener();
    }
    if (!preference.enabled || target.busy || !events.size) return target.busy || Promise.resolve();
    const currentRevision = revision;
    target.busy = Promise.resolve().then(async () => {
      try {
        if (!target.session) {
          const started = await post('/api/usage/session', {
            visitor_key: preference.key, consent_version: CONSENT_VERSION,
            cohort: preference.cohort, identity_scope: persistent ? 'browser' : 'session', channel,
          });
          if (currentRevision !== revision || !preference.enabled) return;
          target.session = started;
        }
        while (preference.enabled && currentRevision === revision) {
          const pending = [...events].filter(([key]) => !sent.has(key)).slice(0, 20);
          if (!pending.length) break;
          await post('/api/usage/events', { token: target.session.token, events: pending.map(([, event]) => event) });
          if (currentRevision !== revision) return;
          for (const [key] of pending) sent.add(key);
        }
        lastError = '';
      } catch (error) { lastError = error.message; }
      finally { active.delete(target.busy); target.busy = null; listener(); }
    });
    active.add(target.busy);
    return target.busy;
  };
  const controls = {
    storageKey,
    beginRun() { run = makeRun(); },
    track(event, { format = 'unknown', detail = '' } = {}) {
      const { events } = run;
      const key = `${event}:${detail}`;
      if (events.size >= 32 && !events.has(key)) return;
      events.set(key, { event, detail, format });
      void flush();
    },
    async setEnabled(value) {
      revision += 1;
      // Finish an already-sent request before deleting or changing identity.
      if (!value) preference.enabled = false;
      await Promise.all([...active]);
      preference.enabled = Boolean(value);
      if (value && (!preference.key || preference.expires <= now())) {
        preference.key = btoa(String.fromCharCode(...crypto.getRandomValues(new Uint8Array(32)))).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
        preference.expires = now() + LIFETIME;
        run.session = null; run.sent.clear();
      }
      preference.version = CONSENT_VERSION;
      persistent = remember();
      listener();
      if (value) await flush();
    },
    async setCohort(cohort) {
      if (!['public', 'test'].includes(cohort) || cohort === preference.cohort) return;
      const wasEnabled = preference.enabled;
      await controls.setEnabled(false);
      if (preference.key) {
        try { await post('/api/usage/cohort', { visitor_key: preference.key, cohort }); }
        catch { lastError = 'Sharing is off. Test-mode change failed; retry to move retained usage.'; listener(); return; }
      }
      preference.cohort = cohort;
      if (wasEnabled) await controls.setEnabled(true);
      else { remember(); listener(); }
    },
    async forget() {
      await controls.setEnabled(false);
      if (preference.key) {
        try { await post('/api/usage/forget', { visitor_key: preference.key }); }
        catch { lastError = 'Sharing is off. Deletion failed; select Delete my usage again to retry.'; listener(); return false; }
      }
      preference = { enabled: false, cohort: preference.cohort };
      run = makeRun(); lastError = '';
      try { storage?.removeItem(storageKey); } catch { /* Optional storage. */ }
      listener(); return true;
    },
    state() { return { enabled: preference.enabled, cohort: preference.cohort, persistent, hasHistory: Boolean(preference.key), error: lastError }; },
    storageChanged(value) {
      let saved = null; try { saved = JSON.parse(value); } catch { /* Treat corrupt/removed preference as withdrawal. */ }
      if (saved?.enabled && saved.key === preference.key && saved.cohort === preference.cohort) return;
      revision += 1;
      preference = saved?.key === preference.key ? { ...saved, enabled: false } : { enabled: false, cohort: 'public' };
      run.session = null; run.sent.clear(); listener();
    },
    subscribe(callback) { listener = callback; callback(); },
    flush: () => flush(),
  };
  return controls;
}

export function mountUsageControls(container, usage) {
  const section = document.createElement('details');
  section.className = 'privacy';
  const summary = document.createElement('summary'); summary.textContent = 'Optional usage statistics';
  const text = document.createElement('p');
  text.textContent = 'Help measure DEEPBOM use by sharing event counts for this analysis and future visits in this channel (ChatGPT or Web). A separate random browser ID measures repeat visits in each channel. No model files, names, hashes, or conversation text are collected. Records expire after 90 days and are deleted by a daily cleanup; aggregate statistics may be published. Analysis works with sharing off.';
  const allow = document.createElement('input'); allow.type = 'checkbox'; allow.dataset.action = 'usage-consent';
  const allowLabel = document.createElement('label'); allowLabel.append(allow, ' Share usage statistics');
  const test = document.createElement('input'); test.type = 'checkbox'; test.dataset.action = 'usage-test';
  const testLabel = document.createElement('label'); testLabel.append(test, ' This browser is for testing (also reclassify its retained usage)');
  const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = 'Delete my usage'; remove.dataset.action = 'usage-forget';
  const note = document.createElement('p'); note.setAttribute('role', 'status');
  const policy = document.createElement('a'); policy.href = 'https://deepbom.org/privacy'; policy.target = '_blank'; policy.rel = 'noopener'; policy.textContent = 'Privacy details';
  section.append(summary, text, allowLabel, document.createElement('br'), testLabel, document.createElement('br'), remove, ' ', policy, note);
  container.append(section);
  let updating = false;
  const update = () => {
    const state = usage.state(); allow.checked = state.enabled; test.checked = state.cohort === 'test';
    allow.disabled = test.disabled = updating; remove.disabled = updating || !state.hasHistory;
    note.textContent = state.error || (state.enabled ? state.persistent ? 'Sharing is on. You can stop or delete this browser’s retained usage at any time.' : 'Sharing this run only. Browser storage is unavailable, so this run is excluded from returning-browser counts.' : 'Sharing is off.');
  };
  const act = async (callback) => { updating = true; update(); try { await callback(); } finally { updating = false; update(); } };
  allow.addEventListener('change', () => { const checked = allow.checked; void act(() => usage.setEnabled(checked)); });
  test.addEventListener('change', () => { const cohort = test.checked ? 'test' : 'public'; void act(() => usage.setCohort(cohort)); });
  remove.addEventListener('click', () => { void act(() => usage.forget()); });
  usage.subscribe(update);
  window.addEventListener('storage', (event) => {
    // Another panel may have withdrawn consent. Stop this panel immediately;
    // opting in again remains a deliberate local action.
    if (event.key === usage.storageKey || event.key === null) usage.storageChanged(event.newValue);
  });
}
