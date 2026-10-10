import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { MessageBubble } from '../features/chat/components/MessageBubble';
import { CitationModalProvider, CitationPanelSlot } from '../features/chat/components/CitationModalProvider';
import './chatContextHarness.css';
import '../../../aquillm/aquillm/static/theme.css';

const docA = '11111111-1111-4111-8111-111111111111';
const docB = '22222222-2222-4222-8222-222222222222';
const titles = ['The Hyper Suprime-Cam survey: filters and photometry', 'Photometric redshifts in deep galaxy surveys'];
const content = `Based on the selected evidence, the **z-channel** is a standard filter used in multi-band photometric surveys to capture near-infrared light from galaxies [doc:${docA} chunk:14864]. Redshift is denoted by the variable *z*, and estimates derived from imaging rather than spectroscopy are commonly called “photo-z’s” [doc:${docB} chunk:13951].

The selected evidence does not explain why the photometric channel and cosmological redshift parameter share the letter “z”.

Sources:
- [doc:${docA} chunk:14864]
- [doc:${docB} chunk:13951]`;

document.body.className = 'theme-aquillm_default_light';
document.cookie = 'csrftoken=preview; path=/';
window.pageUrls = { ...window.pageUrls, document: '/document/%(doc_id)s/' };
window.apiUrls = { ...window.apiUrls, api_citation_sources: '/preview/sources', api_chunk_detail: '/preview/chunks/%(chunk_id)s' };
const realFetch = window.fetch.bind(window);
window.fetch = async (input, init) => {
  const url = String(input);
  if (url === '/preview/sources') return Response.json({ sources: [
    { doc_id: docA, chunk_id: 14864, title: titles[0], modality: 'text' },
    { doc_id: docB, chunk_id: 13951, title: titles[1], modality: 'text' },
  ] });
  if (url.startsWith('/preview/chunks/')) {
    const first = url.includes('14864');
    const text = first ? 'The survey observes galaxies in the g, r, i, z, and y filters.'
      : 'Photometric redshifts are estimated from imaging in multiple bands.';
    return Response.json({ content: text, chunk_number: 1, start_position: 0,
      end_position: text.length, start_time: null, modality: 'text', image_url: null,
      document: { id: first ? docA : docB, title: titles[first ? 0 : 1], type: 'RawTextDocument',
        has_pdf: false, source_url: null, full_text: text, text_offset: 0 } });
  }
  return realFetch(input, init);
};

function Preview() {
  const [narrow, setNarrow] = useState(false);
  const [dark, setDark] = useState(false);
  return <main className="p-5">
    <div className="mb-6 flex gap-3 text-xs">
      <button onClick={() => setNarrow(!narrow)}>{narrow ? 'Desktop width' : 'Mobile width'}</button>
      <button onClick={() => { setDark(!dark); document.body.className = dark ? 'theme-aquillm_default_light' : 'theme-aquillm_default_dark'; }}>
        {dark ? 'Light theme' : 'Dark theme'}
      </button>
      <span className="text-text-low_contrast">Sample documents · citation preview</span>
    </div>
    <CitationModalProvider>
      <div className={narrow ? 'max-w-[390px]' : 'max-w-6xl'}>
        <MessageBubble message={{ role: 'assistant', content }} onRate={() => {}} onFeedback={() => {}} />
      </div>
      <CitationPanelSlot />
    </CitationModalProvider>
  </main>;
}
createRoot(document.getElementById('citation-preview-root')!).render(<Preview />);
