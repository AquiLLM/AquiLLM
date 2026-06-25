import type { Collection } from '../../../components/CollectionsTree';

export interface User {
  id: number;
  username: string;
  email: string;
  full_name: string;
}

export interface CollectionPermissions {
  viewers: number[];
  editors: number[];
  admins: number[];
}

export interface UserWithPermission extends User {
  permission: 'VIEW' | 'EDIT' | 'MANAGE';
}

export interface UserManagementModalProps {
  collection: Collection | null;
  isOpen: boolean;
  onClose: () => void;
  onSave: () => void;
}
