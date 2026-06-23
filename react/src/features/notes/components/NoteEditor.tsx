import { useEffect, useMemo, useState } from 'react';
import Editor, { loader } from '@monaco-editor/react';
import * as monaco from 'monaco-editor';
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';

// Wire Monaco's worker once. Markdown highlighting needs only the base editor
// worker — no JSON / CSS / HTML / TS workers — so we keep the bundle small.
declare global {
  // eslint-disable-next-line no-var
  var MonacoEnvironment: monaco.Environment | undefined;
}

if (typeof self !== 'undefined' && !self.MonacoEnvironment) {
  self.MonacoEnvironment = {
    getWorker() {
      return new EditorWorker();
    },
  };
}

// Use the bundled monaco rather than the default CDN loader.
loader.config({ monaco });

const DARK_THEMES = new Set([
  'theme-aquillm_default_dark',
  'theme-aquillm_default_dark_accessible_chat',
]);
const HC_THEMES = new Set(['theme-high_contrast']);

function deriveMonacoTheme(): 'vs' | 'vs-dark' | 'hc-black' {
  if (typeof document === 'undefined') return 'vs-dark';
  const classes = document.body.className.split(/\s+/);
  if (classes.some(c => HC_THEMES.has(c))) return 'hc-black';
  if (classes.some(c => DARK_THEMES.has(c))) return 'vs-dark';
  return 'vs';
}

function useBodyTheme(): 'vs' | 'vs-dark' | 'hc-black' {
  const [theme, setTheme] = useState<'vs' | 'vs-dark' | 'hc-black'>(deriveMonacoTheme());
  useEffect(() => {
    if (typeof document === 'undefined' || typeof MutationObserver === 'undefined') return;
    const obs = new MutationObserver(() => setTheme(deriveMonacoTheme()));
    obs.observe(document.body, { attributes: true, attributeFilter: ['class'] });
    return () => obs.disconnect();
  }, []);
  return theme;
}

export type NoteEditorProps = {
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
};

export default function NoteEditor({ value, onChange, readOnly }: NoteEditorProps) {
  const theme = useBodyTheme();
  const options = useMemo<monaco.editor.IStandaloneEditorConstructionOptions>(
    () => ({
      wordWrap: 'on',
      minimap: { enabled: false },
      lineNumbers: 'on',
      scrollBeyondLastLine: false,
      automaticLayout: true,
      readOnly: !!readOnly,
      fontSize: 14,
      tabSize: 2,
    }),
    [readOnly],
  );

  return (
    <Editor
      height="100%"
      language="markdown"
      value={value}
      theme={theme}
      options={options}
      onChange={v => onChange(v ?? '')}
    />
  );
}
