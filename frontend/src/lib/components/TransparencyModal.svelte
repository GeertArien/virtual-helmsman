<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { fade } from 'svelte/transition';

  /**
   * AI Act Art. 50 transparency gate.
   *
   * Blocks all interaction with the app until the cursist explicitly
   * acknowledges they are talking to an AI system. Deliberately has NO
   * persistence: every page load remounts this component and shows the modal
   * again, because the simulator cabin may see several cursisten per browser
   * per day. No storage, no network -- acknowledgement logging is out of scope.
   */

  /** Fired once, after the user acknowledges. The modal itself stays free of
   *  any network/side-effect code (per the Art. 50 spec: "no network requests
   *  from the modal itself"); the parent decides what acknowledgement triggers
   *  (e.g. best-effort audit logging). */
  let { onAcknowledge = () => {} }: { onAcknowledge?: () => void } = $props();

  // `open` drives rendering; it starts true on every mount (= every page load).
  // Setting it false on acknowledge lets the {#if} run its fade-out transition.
  let open = $state(true);
  // The checkbox must be ticked before the primary button is enabled.
  let checked = $state(false);

  let dialogEl: HTMLDivElement | undefined = $state();
  let checkboxEl: HTMLInputElement | undefined = $state();

  /** Path to the full declaration. Absolute so it resolves the same on every
   *  route (a relative href would break under /documents/[batch_id] etc.).
   *  Served as a static asset by SvelteKit (frontend/static/...). */
  const DECLARATION_URL = '/documentation/transparantieverklaring.md';

  /** Focusable controls inside the dialog, in DOM order, skipping disabled. */
  function focusable(): HTMLElement[] {
    if (!dialogEl) return [];
    const nodes = dialogEl.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled])'
    );
    return Array.from(nodes);
  }

  onMount(() => {
    // Initial focus on the checkbox (per a11y spec).
    checkboxEl?.focus();

    // Belt-and-suspenders trap: if focus ever lands outside the dialog while
    // the gate is open (e.g. a stray programmatic focus), pull it back. Tab
    // itself is handled in onKeydown; this catches everything else.
    const onFocusIn = (e: FocusEvent) => {
      if (!open || !dialogEl) return;
      const target = e.target as Node | null;
      if (target && !dialogEl.contains(target)) {
        e.stopPropagation();
        (focusable()[0] ?? dialogEl).focus();
      }
    };
    document.addEventListener('focusin', onFocusIn, true);
    return () => document.removeEventListener('focusin', onFocusIn, true);
  });

  function onKeydown(e: KeyboardEvent) {
    if (!open) return;

    // Escape must NOT dismiss the gate -- swallow it.
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      return;
    }

    if (e.key !== 'Tab') return;

    // Focus trap: cycle Tab / Shift+Tab among the dialog's own controls only.
    const items = focusable();
    if (items.length === 0) {
      e.preventDefault();
      return;
    }
    const first = items[0];
    const last = items[items.length - 1];
    const active = document.activeElement as HTMLElement | null;

    if (e.shiftKey) {
      if (active === first || !dialogEl?.contains(active)) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (active === last || !dialogEl?.contains(active)) {
        e.preventDefault();
        first.focus();
      }
    }
  }

  async function acknowledge() {
    if (!checked) return;
    // Closing first disables the trap (handlers early-return on !open), so the
    // focus move below isn't yanked back into the fading modal.
    open = false;
    await tick();
    // Prefer the designated primary input, but only if it's enabled -- the
    // chat box can be disabled (e.g. while browser audio is live), and focus()
    // is a no-op on a disabled control. Fall back to the first usable control
    // in <main> so focus always lands somewhere actionable.
    const target =
      document.querySelector<HTMLElement>('[data-primary-input]:not([disabled])') ??
      document.querySelector<HTMLElement>('[data-primary-input]') ??
      document.querySelector<HTMLElement>(
        'main button:not([disabled]), main input:not([disabled]), main textarea:not([disabled])'
      );
    target?.focus();

    // Notify the parent last, so focus has already moved and the gate is gone
    // regardless of what the callback does.
    onAcknowledge();
  }
</script>

{#if open}
  <!-- Backdrop + centering overlay. The backdrop blocks pointer interaction
       with the underlying UI; the focus trap blocks keyboard interaction. -->
  <div
    class="overlay"
    transition:fade={{ duration: 200 }}
    role="presentation"
  >
    <div
      class="card"
      bind:this={dialogEl}
      role="dialog"
      aria-modal="true"
      aria-labelledby="transparency-title"
      aria-describedby="transparency-intro"
      onkeydown={onKeydown}
      tabindex="-1"
    >
      <h1 id="transparency-title">🤖 AI-systeem actief — graag bevestigen vóór gebruik</h1>

      <p id="transparency-intro" class="intro">
        Dit is een AI-systeem dat scheepscommando's en vragen over maritieme
        regelgeving verwerkt.
      </p>

      <section class="block">
        <h2>✅ Wat het doet</h2>
        <p>
          Engelstalige commando's omzetten naar simulator-acties; vragen
          beantwoorden met bronvermelding (PDF + paginanummer); spraak
          transcribereren (audio blijft in-memory).
        </p>
      </section>

      <section class="block">
        <h2>⚠️ Belangrijke beperkingen</h2>
        <p>
          Het kan hallucineren — verifieer kritieke informatie altijd met het
          officiële brondocument. Geen vervanging voor een gecertificeerde
          instructeur. Niet voor operationele zeevaart, alleen
          simulator-training.
        </p>
      </section>

      <section class="block">
        <h2>🔒 Privacy</h2>
        <p>
          Volledig lokaal — geen cloud, geen verwerkersovereenkomst met derden.
          Audio wordt niet bewaard, geen speaker-identification.
        </p>
      </section>

      <label class="ack">
        <input type="checkbox" bind:checked />
        <span>
          Ik begrijp dat dit een AI-systeem is en dat ik de output kritisch
          evalueer.
        </span>
      </label>

      <div class="actions">
        <button
          type="button"
          class="primary"
          onclick={acknowledge}
          disabled={!checked}
          aria-disabled={!checked}
        >
          Akkoord en starten
        </button>
        <a
          class="declaration"
          href={DECLARATION_URL}
          target="_blank"
          rel="noopener noreferrer"
        >
          Volledige verklaring
        </a>
      </div>
    </div>
  </div>
{/if}

<style>
  .overlay {
    position: fixed;
    inset: 0;
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
    background: rgba(0, 0, 0, 0.5);
  }
  .card {
    width: 100%;
    max-width: 640px;
    max-height: calc(100vh - 2rem);
    overflow-y: auto;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem 1.75rem;
    outline: none;
  }
  h1 {
    margin: 0 0 0.75rem;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.25;
  }
  .intro {
    margin: 0 0 1.25rem;
    color: var(--fg);
    font-size: 0.95rem;
    line-height: 1.5;
  }
  .block {
    padding-top: 0.9rem;
    margin-top: 0.9rem;
    border-top: 1px solid var(--border);
  }
  .block h2 {
    margin: 0 0 0.35rem;
    font-size: 0.95rem;
    font-weight: 600;
  }
  .block p {
    margin: 0;
    color: var(--fg-muted);
    font-size: 0.88rem;
    line-height: 1.5;
  }

  .ack {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    margin: 1.4rem 0 1.1rem;
    padding: 0.7rem 0.8rem;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.9rem;
    line-height: 1.45;
    cursor: pointer;
  }
  .ack input {
    margin-top: 0.15rem;
    width: 1.05rem;
    height: 1.05rem;
    flex: 0 0 auto;
    accent-color: var(--accent);
    cursor: pointer;
  }

  .actions {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .primary {
    background: var(--accent);
    color: #08111e;
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 0.55rem 1.1rem;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
  }
  .primary:hover:not(:disabled) {
    filter: brightness(1.1);
  }
  .primary:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .declaration {
    color: var(--fg-muted);
    font-size: 0.8rem;
    text-decoration: underline;
  }
  .declaration:hover {
    color: var(--accent);
  }

  @media (max-width: 520px) {
    .card {
      padding: 1.25rem 1.1rem;
    }
    h1 {
      font-size: 1.3rem;
    }
  }
</style>
