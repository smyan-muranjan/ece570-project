import React, { createContext, useContext, useState, ReactNode } from 'react';
import { PredictionResponse } from '@/services/api';

interface PredictionContextType {
  prediction: PredictionResponse | null;
  setPrediction: (prediction: PredictionResponse | null) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  error: string | null;
  setError: (error: string | null) => void;
}

const PredictionContext = createContext<PredictionContextType | undefined>(undefined);

export function PredictionProvider({ children }: { children: ReactNode }) {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <PredictionContext.Provider
      value={{
        prediction,
        setPrediction,
        isLoading,
        setIsLoading,
        error,
        setError,
      }}
    >
      {children}
    </PredictionContext.Provider>
  );
}

export function usePrediction() {
  const context = useContext(PredictionContext);
  if (context === undefined) {
    throw new Error('usePrediction must be used within a PredictionProvider');
  }
  return context;
}
