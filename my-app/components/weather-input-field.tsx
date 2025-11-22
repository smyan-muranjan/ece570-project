import React from 'react';
import { StyleSheet, View, Text, TextInput } from 'react-native';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { Colors, Typography, Spacing, BorderRadius, Shadows } from '@/constants/theme';
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
    <View 
      style={[
        styles.container,
        Shadows.card,
        { 
          backgroundColor: isDark ? Colors.dark.secondarySystemGroupedBackground : Colors.light.systemBackground,
          borderWidth: isDark ? 0 : StyleSheet.hairlineWidth,
          borderColor: isDark ? 'transparent' : Colors.light.separator,
        }
      ]}
    >
      <View style={styles.content}>
        <View style={styles.header}>
          <Ionicons name={icon} size={20} color={isDark ? Colors.dark.systemBlue : Colors.light.systemBlue} />
          <Text style={[styles.label, { color: isDark ? Colors.dark.label : Colors.light.label }]}>{label}</Text>
        </View>
        
        <View style={styles.inputContainer}>
          <TextInput
            style={[styles.input, { color: isDark ? Colors.dark.label : Colors.light.label }]}
            value={value}
            onChangeText={onChangeText}
            placeholder={placeholder}
            placeholderTextColor={isDark ? Colors.dark.labelTertiary : Colors.light.labelTertiary}
            keyboardType={keyboardType}
          />
          {unit && (
            <Text style={[styles.unit, { color: isDark ? Colors.dark.labelSecondary : Colors.light.labelSecondary }]}>{unit}</Text>
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: BorderRadius.lg,
    overflow: 'hidden',
    marginBottom: Spacing.sm,
  },
  content: {
    padding: Spacing.md,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    marginBottom: Spacing.sm,
  },
  label: {
    ...Typography.subheadline,
    fontWeight: '600',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  input: {
    flex: 1,
    fontSize: 24,
    fontWeight: '500',
    padding: 0,
  },
  unit: {
    fontSize: 20,
    fontWeight: '500',
    marginLeft: Spacing.xs,
  },
});
