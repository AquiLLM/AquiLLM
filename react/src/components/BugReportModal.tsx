import React, { useState } from 'react';
import { getCsrfCookie } from '../main';

interface BugReportModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const BugReportModal: React.FC<BugReportModalProps> = ({ isOpen, onClose }) => {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || submitting) return;

    setSubmitting(true);
    try {
      const resp = await fetch(window.apiUrls.api_bug_reports, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfCookie(),
        },
        body: JSON.stringify({
          title: title.trim(),
          description: description.trim(),
          url: window.location.href,
          user_agent: navigator.userAgent,
        }),
      });
      if (resp.ok) {
        setSubmitted(true);
        setTimeout(() => {
          setTitle('');
          setDescription('');
          setSubmitted(false);
          onClose();
        }, 1500);
      }
    } catch {
      // silently fail — the user can retry
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 backdrop-blur-[10px] flex justify-center items-center z-[1000]">
      <div className="bg-scheme-shade_3 p-6 rounded-[32px] border border-border-mid_contrast w-full max-w-[440px] relative shadow-lg">
        {submitted ? (
          <div className="text-center py-8">
            <p className="text-text-normal text-lg font-bold">Bug report submitted</p>
            <p className="text-text-low_contrast mt-2">Thank you for your feedback.</p>
          </div>
        ) : (
          <>
            <h3 className="text-2xl font-bold mb-6 text-text-normal">Report a Bug</h3>
            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div>
                <label htmlFor="bugTitle" className="block mb-2 text-text-low_contrast">
                  Title
                </label>
                <input
                  id="bugTitle"
                  type="text"
                  value={title}
                  className="w-full p-3 bg-scheme-shade_4 border border-border-mid_contrast rounded-md text-base text-text-normal focus:outline-none focus:ring-2 focus:ring-accent"
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Brief summary of the issue"
                  autoFocus
                />
              </div>
              <div>
                <label htmlFor="bugDescription" className="block mb-2 text-text-low_contrast">
                  Description
                </label>
                <textarea
                  id="bugDescription"
                  value={description}
                  className="w-full p-3 bg-scheme-shade_4 border border-border-mid_contrast rounded-md text-base text-text-normal focus:outline-none focus:ring-2 focus:ring-accent resize-y min-h-[100px]"
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="What happened? What did you expect?"
                  rows={4}
                />
              </div>
              <div className="flex justify-end gap-3 mt-4">
                <button
                  type="button"
                  onClick={onClose}
                  className="py-2 px-4 rounded bg-scheme-shade_6 hover:bg-scheme-shade_7 text-text-normal text-sm font-medium cursor-pointer border-none"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submitting || !title.trim()}
                  className="py-2 px-4 rounded bg-accent hover:bg-accent-dark text-text-normal text-sm font-medium cursor-pointer border-none disabled:opacity-50"
                >
                  {submitting ? 'Submitting...' : 'Submit'}
                </button>
              </div>
            </form>
          </>
        )}
      </div>
    </div>
  );
};

export default BugReportModal;
