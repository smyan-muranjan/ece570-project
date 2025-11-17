import React from 'react';
import { StyleSheet, View, Text, TextInput, Pressable } from 'react-native';
import { BlurView } from 'expo-blur';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { Ionicons } from '@expo/vector-icons';

interface WeatherInputFieldProps {
  label: string;
  value: string;
  onChangeText: (text: string) => void;
  placeholder: string;
  icon: keyof typeof Ionicons.glyphMap;
  unit?: string;
  keyboardType?: 'default' | 'numeric' | 'decimal-pad';
}

export function WeatherInputField({
  label,
  value,
  onChangeText,
  placeholder,
  icon,
  unit,
  keyboardType = 'decimal-pad'
}: WeatherInputFieldProps) {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  return (
    <BlurView 
      intensity={isDark ? 20 : 80} 
      tint={isDark ? 'dark' : 'light'}
      style={styles.container}
    >
      <View style={styles.content}>
        <View style={styles.header}>
          <Ionicons name={icon} size={20} color={isDark ? '#FFF' : '#007AFF'} />
          <Text style={[styles.label, isDark && styles.labelDark]}>{label}</Text>
        </View>
        
        <View style={styles.inputContainer}>
          <TextInput
            style={[styles.input, isDark && styles.inputDark]}
            value={value}
            onChangeText={onChangeText}
            placeholder={placeholder}
            placeholderTextColor="#8E8E93"
            keyboardType={keyboardType}
          />
          {unit && (
            <Text style={[styles.unit, isDark && styles.unitDark]}>{unit}</Text>
          )}
        </View>
      </View>
    </BlurView>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    marginBottom: 12,
  },
  content: {
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#000',
  },
  labelDark: {
    color: '#FFF',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  input: {
    flex: 1,
    fontSize: 24,
    fontWeight: '500',
    color: '#000',
    padding: 0,
  },
  inputDark: {
    color: '#FFF',
  },
  unit: {
    fontSize: 20,
    fontWeight: '500',
    color: '#8E8E93',
    marginLeft: 8,
  },
  unitDark: {
    color: '#8E8E93',
  },
});
