// Monaco side-by-side diff editor for reviewing a SkillEditSuggestion.
// Left pane = current notes (read-only); right pane = proposed notes (editable).
import { useEffect, useMemo, useState } from 'react';
import { DiffEditor, loader } from '@monaco-editor/react';
import * as monaco from 'monaco-editor';
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';

// Idempotent module-level wiring — safe even if SkillEditor.tsx already ran it.
declare global {
  // eslint-disable-next-line no-var
  var MonacoEnvironment: monaco.Environment | undefined;
}
if (typeof self !== 'undefined' && !self.MonacoEnvironment) {
  self.MonacoEnvironment = { getWorker: () => new EditorWorker() };
}
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

export interface NotesDiffEditorProps {
  original: string;
  modified: string;
  onModifiedChange: (next: string) => void;
}

export default function NotesDiffEditor({ original, modified, onModifiedChange }: NotesDiffEditorProps) {
  const theme = useBodyTheme();
  const options = useMemo<monaco.editor.IDiffEditorConstructionOptions>(
    () => ({
      renderSideBySide: true,
      originalEditable: false,
      readOnly: false,
      wordWrap: 'on',
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      automaticLayout: true,
      fontSize: 14,
      tabSize: 2,
    }),
    [],
  );

  return (
    <DiffEditor
      height="100%"
      language="markdown"
      original={original}
      modified={modified}
      theme={theme}
      options={options}
      onMount={(diffEditor) => {
        diffEditor.getModifiedEditor().onDidChangeModelContent(() => {
          onModifiedChange(diffEditor.getModifiedEditor().getValue());
        });
      }}
    />
  );
}
