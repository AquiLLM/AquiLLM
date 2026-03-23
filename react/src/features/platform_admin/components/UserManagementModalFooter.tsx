import React from 'react';

interface UserManagementModalFooterProps {
  onClose: () => void;
  onSave: () => void;
  isSaving: boolean;
  isLoading: boolean;
}

const UserManagementModalFooter: React.FC<UserManagementModalFooterProps> = ({
  onClose,
  onSave,
  isSaving,
  isLoading,
}) => (
  <div className="border-t border-border-low_contrast px-6 py-4 flex justify-end space-x-3">
    <button
      type="button"
      onClick={onClose}
      className="px-4 py-2 text-text-normal bg-scheme-shade_6 hover:bg-scheme-shade_7 rounded focus:outline-none"
      disabled={isSaving}
    >
      Cancel
    </button>
    <button
      type="button"
      onClick={onSave}
      disabled={isSaving || isLoading}
      className="px-4 py-2 text-text-normal bg-accent hover:bg-accent-dark rounded focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {isSaving ? 'Saving...' : 'Save Changes'}
    </button>
  </div>
);

export default UserManagementModalFooter;
