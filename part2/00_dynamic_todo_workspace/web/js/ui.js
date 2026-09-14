/* Toasts, modals, and focus management. */

import { h, icon } from './dom.js';

const toastHost = () => document.getElementById('toasts');
const live = () => document.getElementById('srStatus');

export function announce(message) {
  const el = live();
  if (el) { el.textContent = ''; setTimeout(() => { el.textContent = message; }, 30); }
}

export function toast({ message, action, tone = 'default', duration = 6000 }) {
  const host = toastHost();
  const el = h('div', { class: `toast ${tone === 'error' ? 'error' : ''}` },
    h('span', { class: 'msg' }, message));

  let timer;
  const dismiss = () => {
    clearTimeout(timer);
    el.classList.add('out');
    el.addEventListener('animationend', () => el.remove(), { once: true });
    setTimeout(() => el.remove(), 400);
  };
  if (action) {
    el.append(h('button', {
      type: 'button',
      onclick: () => { dismiss(); action.onClick(); },
    }, action.label));
  }
  el.append(h('button', { type: 'button', 'aria-label': 'Dismiss', onclick: dismiss,
    style: 'background:none;padding:4px' }, icon('close')));

  // Keep the stack short so it never covers the list.
  [...host.children].slice(0, -2).forEach((c) => c.remove());
  host.append(el);
  announce(message);
  timer = setTimeout(dismiss, duration);
  return dismiss;
}

/* ── modal ───────────────────────────────────────────────────────────── */
let closeCurrent = null;

export function openModal({ title, lede, body, actions = [], onClose }) {
  closeCurrent?.();
  const backdrop = document.getElementById('modalBackdrop');
  const modal = document.getElementById('modal');
  const previous = document.activeElement;

  modal.replaceChildren(
    title ? h('h2', {}, title) : null,
    lede ? h('p', { class: 'lede' }, lede) : null,
    body || null,
    actions.length
      ? h('div', { class: 'modal-actions' }, actions.map((a) =>
        h('button', {
          type: a.type || 'button',
          class: `btn ${a.variant || ''}`,
          onclick: () => { if (a.close !== false) close(); a.onClick?.(); },
        }, a.label)))
      : null,
  );
  backdrop.hidden = false;

  const onKey = (e) => {
    if (e.key === 'Escape') { e.stopPropagation(); close(); }
    if (e.key === 'Tab') trapFocus(e, modal);
  };
  const onClick = (e) => { if (e.target === backdrop) close(); };
  document.addEventListener('keydown', onKey, true);
  backdrop.addEventListener('mousedown', onClick);

  function close() {
    document.removeEventListener('keydown', onKey, true);
    backdrop.removeEventListener('mousedown', onClick);
    backdrop.hidden = true;
    modal.replaceChildren();
    closeCurrent = null;
    onClose?.();
    previous?.focus?.();
  }
  closeCurrent = close;

  // A timer, not requestAnimationFrame: rAF is paused in a background tab and
  // the dialog would open without focus.
  setTimeout(() => {
    (modal.querySelector('input, textarea, button.primary') || modal.querySelector('button'))?.focus();
  }, 0);
  return close;
}

export const modalOpen = () => !document.getElementById('modalBackdrop').hidden;

function trapFocus(e, container) {
  const focusables = container.querySelectorAll(
    'a[href], button:not([disabled]), input, textarea, select, [tabindex]:not([tabindex="-1"])');
  if (!focusables.length) return;
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

export function confirmDialog({ title, lede, confirmLabel = 'Delete', danger = true, onConfirm }) {
  openModal({
    title, lede,
    actions: [
      { label: 'Cancel', variant: 'ghost' },
      { label: confirmLabel, variant: danger ? 'primary' : 'primary', onClick: onConfirm },
    ],
  });
}
