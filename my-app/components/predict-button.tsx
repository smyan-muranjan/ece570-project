import React from 'react';
import { StyleSheet, View, Text, Pressable } from 'react-native';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { Colors, Typography, Spacing, BorderRadius, Shadows } from '@/constants/theme';
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
      <View
        style={[
          styles.gradient,
          Shadows.elevated,
          {
            backgroundColor: disabled || loading 
              ? (isDark ? Colors.dark.labelTertiary : Colors.light.labelTertiary)
              : (isDark ? Colors.dark.systemBlue : Colors.light.systemBlue)
          }
        ]}
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
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: BorderRadius.lg,
    overflow: 'hidden',
  },
  gradient: {
    paddingVertical: Spacing.md + 2,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
  },
  text: {
    ...Typography.headline,
    color: '#FFF',
  },
  disabled: {
    opacity: 0.5,
  },
  pressed: {
    opacity: 0.8,
  },
});
