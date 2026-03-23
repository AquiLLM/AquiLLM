import React, { useState, useEffect } from 'react';
import { FileDown, Trash2 } from 'lucide-react';
import formatUrl from '../utils/formatUrl';
import { getCsrfCookie } from '../main';

export interface WhitelistEmailsProps {
  /** When true, show one-click CSV download (server still enforces superuser on the API). */
  isSuperuser?: boolean;
}

const WhitelistEmails: React.FC<WhitelistEmailsProps> = ({ isSuperuser = false }) => {
  const [emails, setEmails] = useState<string[]>([]);
  const [newEmail, setNewEmail] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [inputError, setInputError] = useState<string | null>(null);

  // Fetch emails on component mount
  useEffect(() => {
    const fetchEmails = async () => {
      try {
        setLoading(true);
        const response = await fetch(window.apiUrls.api_whitelist_emails);
        if (!response.ok) {
          throw new Error('Failed to fetch emails');
        }
        const data: string[] = (await response.json()).whitelisted;
        setEmails(data);
      } catch (err) {
        console.error(err);
        setError('Error fetching emails');
      } finally {
        setLoading(false);
      }
    };

    fetchEmails();
  }, []);

  // Reset input error when newEmail changes
  useEffect(() => {
    setInputError(null);
  }, [newEmail]);

  // Validate email format
  const isValidEmail = (email: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  // Add a new email
  const handleAddEmail = async () => {
    const trimmedEmail = newEmail.trim();
    if (!isValidEmail(trimmedEmail)) {
      setInputError('Invalid email address');
      return;
    }
    if (emails.includes(trimmedEmail)) {
      setInputError('Email already whitelisted');
      return;
    }
    setInputError(null);
    if (trimmedEmail) {
      try {
        setLoading(true);
        const response = await fetch(formatUrl(window.apiUrls.api_whitelist_email, { email: trimmedEmail }), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfCookie()
          },
        });
        if (!response.ok) {
          throw new Error('Failed to add email');
        }
        setEmails(prev => [...prev, trimmedEmail]);
        setNewEmail('');
      } catch (err) {
        console.error(err);
        setError('Error adding email');
      } finally {
        setLoading(false);
      }
    }
  };

  // Delete an email by email
  const handleDelete = async (email: string) => {
    try {
      setLoading(true);
      const response = await fetch(formatUrl(window.apiUrls.api_whitelist_email, { email }), {
        method: 'DELETE',
        headers: {
          'X-CSRFToken': getCsrfCookie()
        },
      });
      if (!response.ok) {
        throw new Error('Failed to delete email');
      }
      setEmails(prev => prev.filter(e => e !== email));
    } catch (err) {
      console.error(err);
      setError('Error deleting email');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleAddEmail();
    }
  };

  const csvUrl = window.apiUrls?.api_feedback_ratings_csv;

  return (
    <div className="w-full p-4 md:px-8">
      <div className="relative mb-6 min-h-[56px] w-full">
        <h2 className="text-center text-xl font-bold text-text-normal">Whitelisted Emails</h2>
        {isSuperuser && csvUrl && (
          <div className="mt-3 flex justify-end sm:absolute sm:right-0 sm:top-1/2 sm:mt-0 sm:-translate-y-1/2">
            <a
              href={csvUrl}
              className="inline-flex h-[56px] w-max shrink-0 cursor-pointer items-center gap-2 rounded-[10px] border border-border-high_contrast bg-scheme-shade_4 px-[16px] text-base text-text-normal no-underline transition-colors duration-200 hover:border-border-higher_contrast hover:bg-scheme-shade_5"
              data-testid="download-feedback-csv"
            >
              <FileDown size={16} className="shrink-0 text-text-normal" aria-hidden />
              Download Feedback CSV
            </a>
          </div>
        )}
      </div>
      <div className="mx-auto max-w-md">
        {error && <div className="text-red-500 mb-4">{error}</div>}
        {loading && <div>Loading...</div>}
        <ul className="list-none p-0">
          {emails.map(email => (
            <li
              key={email}
              className="flex items-center py-2 border-b border-gray-300"
            >
              <span className="flex-grow">{email}</span>
              <button
                onClick={() => handleDelete(email)}
                className="bg-scheme-shade_3 border-none cursor-pointer p-0"
                aria-label={`Delete ${email}`}
              >
                <Trash2 size={18} />
              </button>
            </li>
          ))}
        </ul>
        <div className="mt-4 flex">
          <input
            type="email"
            value={newEmail}
            onChange={(e) => setNewEmail(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Add new email"
            className="flex-grow p-2 bg-scheme-shade_3 text-base border border-gray-300 rounded"
          />
          <button
            onClick={handleAddEmail}
            className="ml-2 p-2 text-base cursor-pointer border border-gray-300 rounded bg-scheme-shade_3"
            disabled={!newEmail.trim() || inputError !== null}
          >
            Add
          </button>
        </div>
        {inputError && <div className="text-red-500 mt-2">{inputError}</div>}
      </div>
    </div>
  );
};

export default WhitelistEmails;
