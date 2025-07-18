import React from 'react';
import { TestComponentProps } from '../types';

const TestComponent: React.FC<TestComponentProps> = ({ message = 'Default message' }) => {
  return (
    <div>
      <h2>{message}</h2>
    </div>
  );
};

export default TestComponent;