import { createRoot } from 'react-dom/client';
import Chat from '../features/chat/components/Chat';
import type { ChatContextCatalog, SkillOverrides } from '../features/chat/types';
import './chatContextHarness.css';
import '../../../aquillm/aquillm/static/theme.css';

const catalog: ChatContextCatalog = {
  skills_enabled: true,
  collections: [
    { id: 'test2', name: 'Test 2', parent: null, path: 'Test 2', is_skill_pack: false },
    { id: 'figures', name: '2504.19874v1 - Figures', parent: 'test2', path: 'Test 2 / 2504.19874v1 - Figures', is_skill_pack: false },
    { id: 'pack', name: 'skill_pack', parent: 'test2', path: 'Test 2 / skill_pack', is_skill_pack: true },
    { id: 'example', name: 'Example Collection', parent: null, path: 'Example Collection', is_skill_pack: false },
    { id: 'kg', name: 'KG Test', parent: null, path: 'KG Test', is_skill_pack: false },
  ],
  skills: [
    { id: 'raw:1', name: 'Cite sources', description: 'Connect factual claims to source documents.', instructions: 'Support factual claims with citations to the supplied documents.\nIf the available sources do not support a claim, say so.', collection_id: 'test2', collection_name: 'Test 2', collection_path: 'Test 2', source_path: 'Test 2 / citations_skill.md', pack_id: null, pack_name: null, default_collection_ids: ['test2'] },
    { id: 'raw:2', name: 'Separate evidence from inference', description: 'Make interpretation and uncertainty clear.', instructions: 'Clearly distinguish source evidence from your own interpretation. State uncertainty where the evidence is incomplete.', collection_id: 'test2', collection_name: 'Test 2', collection_path: 'Test 2', source_path: 'Test 2 / evidence_skill.md', pack_id: null, pack_name: null, default_collection_ids: ['test2'] },
    { id: 'raw:3', name: 'Concise answers', description: 'Lead with the answer. Keep the detail focused.', instructions: 'Start with the answer, then add only the supporting detail needed to make it useful.', collection_id: 'pack', collection_name: 'skill_pack', collection_path: 'Test 2 / skill_pack', source_path: 'Test 2 / skill_pack / concise.md', pack_id: 'pack', pack_name: 'skill_pack', default_collection_ids: ['test2', 'pack'] },
    { id: 'raw:4', name: 'Compare sources', description: 'Highlight agreement, differences, and gaps.', instructions: 'When multiple sources address the same question, identify where they agree and where they differ.', collection_id: 'pack', collection_name: 'skill_pack', collection_path: 'Test 2 / skill_pack', source_path: 'Test 2 / skill_pack / compare.md', pack_id: 'pack', pack_name: 'skill_pack', default_collection_ids: ['test2', 'pack'] },
  ],
};
const parameters = new URLSearchParams(location.search);
document.body.className = parameters.has('dark') ? 'theme-aquillm_default_dark' : 'theme-aquillm_default_light';
if (parameters.has('disabled')) { catalog.skills_enabled = false; catalog.skills = []; }
let saved: { selected_collections: string[]; skill_overrides: SkillOverrides } = JSON.parse(sessionStorage.getItem('context-harness') || 'null') ?? {
  selected_collections: ['test2', 'figures'], skill_overrides: { 'raw:3': true, 'raw:4': false },
};
const realFetch = window.fetch;
window.fetch = (input, init) => String(input) === '/api/collections/chat-context/'
  ? Promise.resolve(new Response(JSON.stringify(catalog), { status: 200, headers: { 'Content-Type': 'application/json' } }))
  : realFetch(input, init);
class HarnessSocket {
  static OPEN = 1; static CLOSED = 3;
  readyState = 1;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor() { setTimeout(() => { this.onopen?.(); this.emit({ conversation: { messages: [], ...saved } }); }, 20); }
  emit(payload: unknown) { this.onmessage?.({ data: JSON.stringify(payload) }); }
  close() { this.readyState = 3; }
  send(raw: string) {
    const request = JSON.parse(raw);
    if (request.action === 'select_collections') {
      saved = { selected_collections: request.collections, skill_overrides: request.skill_overrides ?? saved.skill_overrides };
      sessionStorage.setItem('context-harness', JSON.stringify(saved));
      setTimeout(() => this.emit({ context_selection: { ...saved, request_id: request.request_id } }), 150);
    }
    if (request.action === 'append') setTimeout(() => this.emit({ delta: { messages: [{ role: 'assistant', content: 'Your applied collections and skills are ready for this chat.', message_uuid: 'reply' }] } }), 100);
  }
}
window.WebSocket = HarnessSocket as unknown as typeof WebSocket;
createRoot(document.getElementById('chat-context-root')!).render(<Chat convoId="context-preview" />);
