/**
 * Pollen Predictor - Apple Weather-inspired Design System
 * Beautiful gradients and glassmorphism effects
 */

import { Platform } from 'react-native';

const tintColorLight = '#007AFF';
const tintColorDark = '#0A84FF';

// Pollen severity color scale
export const PollenColors = {
  none: '#34C759',        // Green
  veryLow: '#30D158',     // Bright Green
  low: '#64D2FF',         // Light Blue
  lowMod: '#5AC8FA',      // Sky Blue
  moderate: '#FFD60A',    // Yellow
  modHigh: '#FF9F0A',     // Orange
  high: '#FF9500',        // Deep Orange
  veryHigh: '#FF6482',    // Pink-Red
  extreme: '#FF453A',     // Red
  severe: '#BF5AF2',      // Purple
};

// Allergen type colors
export const AllergenColors = {
  tree: '#34C759',        // Green
  grass: '#FFD60A',       // Yellow
  ragweed: '#FF9F0A',     // Orange
  weed: '#BF5AF2',        // Purple
};

// Gradient backgrounds for different pollen levels
export const PollenGradients = {
  none: ['#D4F1F4', '#A7E6EA'],
  low: ['#BFE3FF', '#8AC8FF'],
  moderate: ['#FFE5A0', '#FFCC5C'],
  high: ['#FFB88C', '#FF9A5A'],
  veryHigh: ['#FF8FA3', '#FF6B7A'],
  severe: ['#E0B4FF', '#BF5AF2'],
};

export const Colors = {
  light: {
    text: '#000000',
    textSecondary: '#8E8E93',
    background: '#F2F2F7',
    backgroundElevated: '#FFFFFF',
    tint: tintColorLight,
    icon: '#8E8E93',
    tabIconDefault: '#8E8E93',
    tabIconSelected: tintColorLight,
    border: '#C6C6C8',
    cardBackground: 'rgba(255, 255, 255, 0.85)',
    shadowColor: 'rgba(0, 0, 0, 0.1)',
  },
  dark: {
    text: '#FFFFFF',
    textSecondary: '#8E8E93',
    background: '#000000',
    backgroundElevated: '#1C1C1E',
    tint: tintColorDark,
    icon: '#8E8E93',
    tabIconDefault: '#8E8E93',
    tabIconSelected: tintColorDark,
    border: '#38383A',
    cardBackground: 'rgba(28, 28, 30, 0.85)',
    shadowColor: 'rgba(0, 0, 0, 0.3)',
  },
};

export const Fonts = Platform.select({
  ios: {
    /** iOS `UIFontDescriptorSystemDesignDefault` */
    sans: 'system-ui',
    /** iOS `UIFontDescriptorSystemDesignSerif` */
    serif: 'ui-serif',
    /** iOS `UIFontDescriptorSystemDesignRounded` */
    rounded: 'ui-rounded',
    /** iOS `UIFontDescriptorSystemDesignMonospaced` */
    mono: 'ui-monospace',
  },
  default: {
    sans: 'normal',
    serif: 'serif',
    rounded: 'normal',
    mono: 'monospace',
  },
  web: {
    sans: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    serif: "Georgia, 'Times New Roman', serif",
    rounded: "'SF Pro Rounded', 'Hiragino Maru Gothic ProN', Meiryo, 'MS PGothic', sans-serif",
    mono: "SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
  },
});
