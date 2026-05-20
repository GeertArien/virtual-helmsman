/** Flat log-row type shared between the page (which builds entries) and the
 *  ConversationPanel (which renders them). Lives in a .ts file because Svelte
 *  components do not re-export module-level types cleanly. */
export type Entry =
  | { kind: 'user'; ts: string; text: string }
  | { kind: 'assistant'; ts: string; text: string }
  | { kind: 'action'; ts: string; label: string; ok: boolean }
  | { kind: 'refused'; ts: string; reason: string };
