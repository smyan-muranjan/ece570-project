import React from 'react';
import { StyleSheet, View, Text, Pressable } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { Ionicons } from '@expo/vector-icons';

interface PredictButtonProps {
  onPress: () => void;
  disabled?: boolean;
  loading?: boolean;
}

export function PredictButton({ onPress, disabled = false, loading = false }: PredictButtonProps) {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  return (
    <Pressable
      onPress={onPress}
      disabled={disabled || loading}
      style={({ pressed }) => [
        styles.container,
        (disabled || loading) && styles.disabled,
        pressed && styles.pressed,
      ]}
    >
      <LinearGradient
        colors={disabled || loading 
          ? ['#8E8E93', '#8E8E93'] as const
          : isDark
          ? ['#0A84FF', '#0066CC'] as const
          : ['#007AFF', '#0051D5'] as const
        }
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
        style={styles.gradient}
      >
        <View style={styles.content}>
          {loading ? (
            <>
              <Ionicons name="hourglass-outline" size={24} color="#FFF" />
              <Text style={styles.text}>Predicting...</Text>
            </>
          ) : (
            <>
              <Ionicons name="analytics" size={24} color="#FFF" />
              <Text style={styles.text}>Predict Pollen Levels</Text>
            </>
          )}
        </View>
      </LinearGradient>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 16,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 12,
    elevation: 5,
  },
  gradient: {
    paddingVertical: 18,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  text: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFF',
  },
  disabled: {
    opacity: 0.5,
  },
  pressed: {
    opacity: 0.8,
  },
});
