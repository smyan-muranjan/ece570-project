import React from 'react';
import { StyleSheet, View, ViewStyle } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { useColorScheme } from '@/hooks/use-color-scheme';

interface GradientCardProps {
  children: React.ReactNode;
  colors?: readonly [string, string, ...string[]];
  style?: ViewStyle;
  intensity?: number;
}

export function GradientCard({ 
  children, 
  colors = ['rgba(255, 255, 255, 0.9)', 'rgba(255, 255, 255, 0.7)'] as const,
  style,
  intensity = 80
}: GradientCardProps) {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  return (
    <View style={[styles.container, style]}>
      <BlurView 
        intensity={intensity} 
        tint={isDark ? 'dark' : 'light'}
        style={styles.blurContainer}
      >
        <LinearGradient
          colors={isDark 
            ? ['rgba(28, 28, 30, 0.95)', 'rgba(28, 28, 30, 0.8)'] as const
            : colors
          }
          style={styles.gradient}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        >
          {children}
        </LinearGradient>
      </BlurView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 20,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 5,
  },
  blurContainer: {
    overflow: 'hidden',
    borderRadius: 20,
  },
  gradient: {
    padding: 20,
  },
});
