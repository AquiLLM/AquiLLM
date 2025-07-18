import React, { ChangeEvent, useState } from 'react';
import { getCsrfCookie } from '../main';

// Helper component to render a text chunk, similar to text_chunk.html
interface TextChunkProps {
  item: {
    document: {
      title: string;
    };
    start_position: number;
    end_position: number;
    content: string;
  };
}

const TextChunk: React.FC<TextChunkProps> = ({ item }) => (
  <details>
    <summary>
      {item.document.title} {item.start_position} → {item.end_position}
    </summary>
    <p>{item.content}</p>
  </details>
);

interface Collection {
  id: string | number;
  name: string;
}

const SearchPage: React.FC<{ availableCollections: Collection[] }> = ({ availableCollections = [] }) => {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [collections, setCollections] = useState<(string | number)[]>([]); // array of selected collection ids
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [rerankedResults, setRerankedResults] = useState([]);
  const [vectorResults, setVectorResults] = useState([]);
  const [trigramResults, setTrigramResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setErrorMessage(null);

    try {
      const response = await fetch('/search/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfCookie(),
        },
        body: JSON.stringify({
          query,
          top_k: topK,
          collections,
        }),
      });
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();

      if (data.error_message) {
        setErrorMessage(data.error_message);
      } else {
        setRerankedResults(data.reranked_results || []);
        setVectorResults(data.vector_results || []);
        setTrigramResults(data.trigram_results || []);
      }
    } catch (error) {
      if (error instanceof Error) {
        setErrorMessage(error.message || 'Something went wrong.');
      } else {
        setErrorMessage('Something went wrong.');
      }
    }
    setLoading(false);
  };

  // Handler for checkbox changes
  const handleCollectionChange = (collId: string | number, checked: boolean) => {
    setCollections((prev) =>
      checked ? [...prev, collId] : prev.filter((id) => id !== collId)
    );
  };

  return (
    <div>
      <h1>Search</h1>
      {errorMessage && <div className="error_message">{errorMessage}</div>}
      <form onSubmit={handleSubmit}>
        <div>
          <label>
            Query:
            <textarea
              value={query}
              placeholder="Enter search query..."
              onChange={(e) => setQuery(e.target.value)}
              rows={4}
              className="w-full resize-none p-3 mb-3 rounded-md bg-lightest-primary"
            />
          </label>
        </div>
        <div>
          <label>
            Top K:
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value, 10))}
              min={1}
              max={200}
              className="w-full m-2 p-2 rounded-md bg-lightest-primary"
            />
          </label>
        </div>
        <div>
          <fieldset>
            <legend>Collections:</legend>
            {availableCollections.map((coll) => (
              <label key={coll.id} style={{ display: 'block', marginBottom: '0.5rem' }}>
                <input
                  type="checkbox"
                  value={coll.id}
                  checked={collections.includes(String(coll.id))}
                  onChange={(e) =>
                    handleCollectionChange(String(coll.id), e.target.checked)
                  }
                />
                {coll.name}
              </label>
            ))}
          </fieldset>
        </div>
        <div>
          <input type="submit" value={loading ? "Submitting..." : "Submit"} disabled={loading} />
        </div>
      </form>

      {rerankedResults.length > 0 && (
        <div>
          <h2>Reranked Search Results</h2>
          <ul>
            {rerankedResults.map((item, index) => (
              <li key={`reranked-${index}`}>
                <TextChunk item={item} />
              </li>
            ))}
          </ul>
        </div>
      )}
      {vectorResults.length > 0 && (
        <div>
          <h2>Vector Search Results</h2>
          <ul>
            {vectorResults.map((item, index) => (
              <li key={`vector-${index}`}>
                <TextChunk item={item} />
              </li>
            ))}
          </ul>
        </div>
      )}
      {trigramResults.length > 0 && (
        <div>
          <h2>Trigram Search Results</h2>
          <ul>
            {trigramResults.map((item, index) => (
              <li key={`trigram-${index}`}>
                <TextChunk item={item} />
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default SearchPage;
