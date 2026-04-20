import React from 'react';
import type { Collection } from '../../../components/CollectionsTree';

interface UserManagementModalHeaderHelpProps {
  collection: Collection;
  onClose: () => void;
}

const UserManagementModalHeaderHelp: React.FC<UserManagementModalHeaderHelpProps> = ({
  collection,
  onClose,
}) => (
  <>
    <div className="px-6 py-4 border-b border-border-low_contrast flex justify-between items-center">
      <h2 className="text-xl font-semibold">Manage Collaborators: {collection.name}</h2>
      <button type="button" onClick={onClose} className="text-text-lower_contrast hover:text-text-normal">
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>

    <div className="px-6 py-3 bg-scheme-shade_4 border-b border-border-lower_contrast">
      <div className="flex items-start">
        <svg
          className="w-5 h-5 mt-0.5 mr-2 text-accent"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <div>
          <h3 className="text-sm font-semibold text-accent mb-1">About Permissions</h3>
          <p className="text-xs text-text-low_contrast mb-2">
            Users with permissions on a collection automatically inherit access to all subcollections:
          </p>
          <ul className="text-xs text-text-low_contrast list-disc list-inside ml-2 space-y-1">
            <li>
              <span className="font-semibold">Viewer</span> - Can view documents but cannot make changes
            </li>
            <li>
              <span className="font-semibold">Editor</span> - Can add, edit, and delete documents
            </li>
            <li>
              <span className="font-semibold">Admin</span> - Can manage users and collection settings
            </li>
          </ul>
        </div>
      </div>
    </div>
  </>
);

export default UserManagementModalHeaderHelp;
