<script lang="ts">
  /**
   * Renders a single AppConfig field based on its JSON Schema entry.
   *
   * The widget is picked by inspecting the schema:
   *
   *   enum present              -> <select>
   *   const present             -> <select> (single-value Literal; Pydantic
   *                                 emits ``const`` rather than a one-item
   *                                 enum -- normalised at resolve() time)
   *   type=boolean              -> <input type="checkbox">
   *   type=integer              -> <input type="number" step=1>
   *   type=number               -> <input type="number" step=any>
   *   type=string               -> <input type="text">
   *   type=array                -> comma-separated <input type="text">
   *                                (one-line editor; fine for `cors_allow_origins`)
   *   type=object               -> recursive nested <SchemaField>s for each key
   *   anyOf with null           -> nullable variant of the inner type
   *
   * `$ref` is resolved against `rootSchema.$defs` lazily.
   */
  import type { JsonSchema, JsonSchemaProperty } from '$lib/api';
  import SchemaField from './SchemaField.svelte';

  let {
    name,
    schema,
    value,
    rootSchema,
    path,
    fieldError,
    onChange
  }: {
    name: string;
    schema: JsonSchemaProperty;
    value: unknown;
    rootSchema: JsonSchema;
    /** Pydantic error `loc` path for this field, used to look up errors. */
    path: (string | number)[];
    /** Resolve a Pydantic error loc to a string if any matches this path. */
    fieldError: (loc: (string | number)[]) => string | null;
    onChange: (newValue: unknown) => void;
  } = $props();

  /** Resolve `$ref` once, returning the dereferenced schema. Also
   *  normalises Pydantic's ``const`` (emitted for single-value Literals)
   *  into a one-element ``enum`` so the renderer's existing
   *  enum-as-<select> branch handles it without a special case. */
  function resolve(s: JsonSchemaProperty): JsonSchemaProperty {
    let out = s;
    if (out.$ref) {
      // "#/$defs/Foo" -> rootSchema.$defs.Foo
      const m = /^#\/\$defs\/(.+)$/.exec(out.$ref);
      if (m && rootSchema.$defs && m[1] in rootSchema.$defs) {
        out = rootSchema.$defs[m[1]];
      }
    }
    if (out.const !== undefined && !out.enum) {
      out = { ...out, enum: [out.const] };
    }
    return out;
  }

  /** Pick the "primary" subtype for an anyOf [T, null] union. */
  function unwrapNullable(s: JsonSchemaProperty): { inner: JsonSchemaProperty; nullable: boolean } {
    if (!s.anyOf) return { inner: s, nullable: false };
    const nonNull = s.anyOf.filter((b) => b.type !== 'null');
    const hasNull = s.anyOf.some((b) => b.type === 'null');
    if (nonNull.length === 1) return { inner: nonNull[0], nullable: hasNull };
    return { inner: s, nullable: hasNull };
  }

  const resolved = $derived(resolve(schema));
  const { inner, nullable } = $derived.by(() => unwrapNullable(resolved));
  const innerResolved = $derived(resolve(inner));

  const error = $derived(fieldError(path));

  function emit(v: unknown) {
    onChange(v);
  }

  function setObjectKey(key: string, sub: unknown) {
    const current = (value as Record<string, unknown> | null) ?? {};
    emit({ ...current, [key]: sub });
  }

  /** Display label = JSON Schema `title` if present, else the field name. */
  const label = $derived(innerResolved.title ?? name);

  /** Comma-separated <-> array conversion for the simple list editor. */
  function arrayToString(arr: unknown): string {
    if (Array.isArray(arr)) return arr.map((x) => String(x)).join(', ');
    return '';
  }
  function stringToArray(s: string): string[] {
    return s
      .split(',')
      .map((x) => x.trim())
      .filter((x) => x.length > 0);
  }
</script>

{#if innerResolved.type === 'object' && innerResolved.properties}
  <fieldset class="nested">
    <legend>{label}</legend>
    {#if innerResolved.description}
      <p class="hint">{innerResolved.description}</p>
    {/if}
    {#each Object.entries(innerResolved.properties) as [k, sub] (k)}
      <SchemaField
        name={k}
        schema={sub}
        value={(value as Record<string, unknown> | null)?.[k]}
        {rootSchema}
        path={[...path, k]}
        {fieldError}
        onChange={(v) => setObjectKey(k, v)}
      />
    {/each}
  </fieldset>
{:else}
  <label class="row" class:has-error={error !== null}>
    <span class="lbl">
      <span class="name">{label}</span>
      {#if innerResolved.description}
        <span class="desc" title={innerResolved.description}>?</span>
      {/if}
    </span>

    {#if innerResolved.enum}
      <select
        value={value === null || value === undefined ? '' : (value as string)}
        onchange={(e) => {
          const raw = (e.currentTarget as HTMLSelectElement).value;
          // The (null) option carries value="" so we can host it in the same
          // <select>. Translate that back to JSON `null` on the way out --
          // an empty string would fail any Literal validator on the backend.
          if (raw === '' && nullable) emit(null);
          else emit(raw);
        }}
      >
        {#if nullable}
          <option value="">(null)</option>
        {/if}
        {#each innerResolved.enum as opt (String(opt))}
          <option value={opt as string}>{String(opt)}</option>
        {/each}
      </select>
    {:else if innerResolved.type === 'boolean'}
      <input
        type="checkbox"
        checked={Boolean(value)}
        onchange={(e) => emit((e.currentTarget as HTMLInputElement).checked)}
      />
    {:else if innerResolved.type === 'integer'}
      <input
        type="number"
        step="1"
        value={value === null || value === undefined ? '' : (value as number)}
        oninput={(e) => {
          const raw = (e.currentTarget as HTMLInputElement).value;
          emit(raw === '' ? (nullable ? null : 0) : Math.trunc(Number(raw)));
        }}
      />
    {:else if innerResolved.type === 'number'}
      <input
        type="number"
        step="any"
        value={value === null || value === undefined ? '' : (value as number)}
        oninput={(e) => {
          const raw = (e.currentTarget as HTMLInputElement).value;
          emit(raw === '' ? (nullable ? null : 0) : Number(raw));
        }}
      />
    {:else if innerResolved.type === 'array'}
      <input
        type="text"
        value={arrayToString(value)}
        oninput={(e) => emit(stringToArray((e.currentTarget as HTMLInputElement).value))}
        placeholder="comma-separated"
      />
    {:else}
      <input
        type="text"
        value={value === null || value === undefined ? '' : String(value)}
        oninput={(e) => {
          const raw = (e.currentTarget as HTMLInputElement).value;
          if (raw === '' && nullable) emit(null);
          else emit(raw);
        }}
        placeholder={nullable ? '(null)' : ''}
      />
    {/if}

    {#if error}
      <span class="err">{error}</span>
    {/if}
  </label>
{/if}

<style>
  .row {
    display: grid;
    grid-template-columns: minmax(12rem, 16rem) 1fr;
    gap: 0.6rem 0.75rem;
    align-items: center;
    padding: 0.25rem 0;
  }
  .row.has-error input,
  .row.has-error select {
    border-color: var(--bad);
  }
  .lbl {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: var(--fg);
    font-size: 0.85rem;
  }
  .name {
    font-family: ui-monospace, 'JetBrains Mono', SFMono-Regular, Menlo, monospace;
  }
  .desc {
    width: 1.1rem;
    height: 1.1rem;
    border-radius: 50%;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    color: var(--fg-muted);
    font-size: 0.7rem;
    line-height: 1.1rem;
    text-align: center;
    cursor: help;
  }

  input[type='text'],
  input[type='number'],
  select {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.3rem 0.55rem;
    color: var(--fg);
    font-size: 0.85rem;
    outline: none;
    font-family: ui-monospace, 'JetBrains Mono', SFMono-Regular, Menlo, monospace;
  }
  input:focus,
  select:focus {
    border-color: var(--accent);
  }
  input[type='checkbox'] {
    width: 1rem;
    height: 1rem;
    accent-color: var(--accent);
    justify-self: start;
  }

  .err {
    grid-column: 2;
    color: var(--bad);
    font-size: 0.75rem;
  }

  .nested {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.5rem 0.75rem 0.6rem;
    margin: 0.4rem 0;
    background: var(--bg-elev-2);
  }
  .nested legend {
    color: var(--fg-muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0 0.35rem;
  }
  .hint {
    margin: 0 0 0.4rem 0;
    font-size: 0.78rem;
    color: var(--fg-muted);
  }
</style>
