import React, { useState, useEffect, useRef } from 'react';
import { getCookie } from '../../../utils/csrf';
import formatUrl from '../../../utils/formatUrl';
import UserManagementModalFooter from './UserManagementModalFooter';
import UserManagementModalHeaderHelp from './UserManagementModalHeaderHelp';
import UserManagementModalScrollBody from './UserManagementModalScrollBody';
import type {
  CollectionPermissions,
  User,
  UserManagementModalProps,
  UserWithPermission,
} from './userManagementTypes';

const UserManagementModal: React.FC<UserManagementModalProps> = ({
  collection,
  isOpen,
  onClose,
  onSave,
}) => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<User[]>([]);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [currentUsers, setCurrentUsers] = useState<UserWithPermission[]>([]);
  const [permissions, setPermissions] = useState<CollectionPermissions>({
    viewers: [],
    editors: [],
    admins: [],
  });
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (isOpen) {
      setError(null);
      setSuccessMessage(null);
    }
  }, [isOpen]);

  useEffect(() => {
    if (isOpen && collection) {
      void fetchCurrentUsers();
    }
  }, [isOpen, collection]);

  useEffect(() => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    if (searchQuery.trim().length > 0) {
      searchTimeoutRef.current = setTimeout(() => {
        void searchUsers(searchQuery);
      }, 300);
    } else {
      setSearchResults([]);
    }

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [searchQuery]);

  const fetchCurrentUsers = async () => {
    if (!collection) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        formatUrl(window.apiUrls.api_collection_permissions, { col_id: collection.id }),
        {
          headers: {
            Accept: 'application/json',
            'X-CSRFToken': getCookie('csrftoken'),
          },
          credentials: 'include',
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch current users');
      }

      const data = await response.json();

      const viewers = data.viewers || [];
      const editors = data.editors || [];
      const admins = data.admins || [];

      setPermissions({
        viewers: viewers.map((user: User) => user.id),
        editors: editors.map((user: User) => user.id),
        admins: admins.map((user: User) => user.id),
      });

      const allUsers: UserWithPermission[] = [
        ...viewers.map((user: User) => ({ ...user, permission: 'VIEW' as const })),
        ...editors.map((user: User) => ({ ...user, permission: 'EDIT' as const })),
        ...admins.map((user: User) => ({ ...user, permission: 'MANAGE' as const })),
      ];

      setCurrentUsers(allUsers);
    } catch (err) {
      console.error('Error fetching users:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch current users');
    } finally {
      setIsLoading(false);
    }
  };

  const searchUsers = async (query: string) => {
    setIsSearching(true);
    try {
      const response = await fetch(
        `${window.apiUrls.api_search_users}?query=${encodeURIComponent(query)}&exclude_current=true`,
        {
          headers: {
            Accept: 'application/json',
          },
          credentials: 'include',
        }
      );

      if (!response.ok) {
        throw new Error('Failed to search users');
      }

      const data = await response.json();
      setSearchResults(data.users || []);
    } catch (err) {
      console.error('Error searching users:', err);
      setError('Failed to search users');
    } finally {
      setIsSearching(false);
    }
  };

  const handleSave = async () => {
    if (!collection) return;

    setIsSaving(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const response = await fetch(
        formatUrl(window.apiUrls.api_collection_permissions, { col_id: collection.id }),
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken'),
          },
          body: JSON.stringify(permissions),
          credentials: 'include',
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        let errorMessage = 'Failed to update permissions';
        if (errorData.error && errorData.error.includes('User matching query does not exist')) {
          errorMessage =
            'One or more users could not be found, possibly because they were deleted. Please close and reopen the dialog to refresh the list, or try removing the affected users.';
        } else {
          errorMessage = errorData.error || 'Failed to update permissions';
        }
        throw new Error(errorMessage);
      }

      setSuccessMessage('Permissions updated successfully!');

      setTimeout(() => {
        onSave();
        onClose();
      }, 1500);
    } catch (err) {
      console.error('Error saving permissions:', err);
      let errorMessage = 'Failed to update permissions';
      if (err instanceof Error) {
        if (err.message.includes('User matching query does not exist')) {
          errorMessage =
            'One or more users could not be found, possibly because they were deleted. Please close and reopen the dialog to refresh the list, or try removing the affected users.';
        } else {
          errorMessage = err.message;
        }
      }
      setError(errorMessage);
    } finally {
      setIsSaving(false);
    }
  };

  const addUser = (user: User, permission: 'VIEW' | 'EDIT' | 'MANAGE') => {
    const newPermissions = {
      viewers: permissions.viewers.filter((id) => id !== user.id),
      editors: permissions.editors.filter((id) => id !== user.id),
      admins: permissions.admins.filter((id) => id !== user.id),
    };

    if (permission === 'VIEW') {
      newPermissions.viewers.push(user.id);
    } else if (permission === 'EDIT') {
      newPermissions.editors.push(user.id);
    } else if (permission === 'MANAGE') {
      newPermissions.admins.push(user.id);
    }

    setPermissions(newPermissions);

    const existingUserIndex = currentUsers.findIndex((u) => u.id === user.id);
    if (existingUserIndex >= 0) {
      const updatedUsers = [...currentUsers];
      updatedUsers[existingUserIndex] = { ...updatedUsers[existingUserIndex], permission };
      setCurrentUsers(updatedUsers);
    } else {
      setCurrentUsers([...currentUsers, { ...user, permission }]);
    }

    setSearchResults([]);
    setSearchQuery('');
  };

  const removeUser = (userId: number) => {
    setPermissions({
      viewers: permissions.viewers.filter((id) => id !== userId),
      editors: permissions.editors.filter((id) => id !== userId),
      admins: permissions.admins.filter((id) => id !== userId),
    });

    setCurrentUsers(currentUsers.filter((user) => user.id !== userId));
  };

  const changeUserPermission = (userId: number, newPermission: 'VIEW' | 'EDIT' | 'MANAGE') => {
    const newPermissions = {
      viewers: permissions.viewers.filter((id) => id !== userId),
      editors: permissions.editors.filter((id) => id !== userId),
      admins: permissions.admins.filter((id) => id !== userId),
    };

    if (newPermission === 'VIEW') {
      newPermissions.viewers.push(userId);
    } else if (newPermission === 'EDIT') {
      newPermissions.editors.push(userId);
    } else if (newPermission === 'MANAGE') {
      newPermissions.admins.push(userId);
    }

    setPermissions(newPermissions);

    setCurrentUsers(
      currentUsers.map((user) => (user.id === userId ? { ...user, permission: newPermission } : user))
    );
  };

  if (!isOpen || !collection) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-scheme-shade_3 rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        <UserManagementModalHeaderHelp collection={collection} onClose={onClose} />

        <UserManagementModalScrollBody
          successMessage={successMessage}
          error={error}
          isLoading={isLoading}
          searchQuery={searchQuery}
          onSearchQueryChange={setSearchQuery}
          isSearching={isSearching}
          searchResults={searchResults}
          onAddUser={addUser}
          currentUsers={currentUsers}
          onChangePermission={changeUserPermission}
          onRemoveUser={removeUser}
        />

        <UserManagementModalFooter
          onClose={onClose}
          onSave={() => void handleSave()}
          isSaving={isSaving}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
};

export default UserManagementModal;
