/**
 * Pollen Predictor - Apple Design System
 * Following iOS Human Interface Guidelines
 */

import { Platform } from 'react-native';

// Apple's system colors
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

// Apple's semantic colors following HIG
export const Colors = {
  light: {
    // Labels
    label: '#000000',
    labelSecondary: '#3C3C43',
    labelTertiary: '#3C3C4399',
    labelQuaternary: '#3C3C432E',
    
    // Fills
    fill: '#78788033',
    fillSecondary: '#78788028',
    fillTertiary: '#7676801E',
    fillQuaternary: '#74748014',
    
    // Backgrounds
    systemBackground: '#FFFFFF',
    secondarySystemBackground: '#F2F2F7',
    tertiarySystemBackground: '#FFFFFF',
    
    // Grouped backgrounds
    systemGroupedBackground: '#F2F2F7',
    secondarySystemGroupedBackground: '#FFFFFF',
    tertiarySystemGroupedBackground: '#F2F2F7',
    
    // System colors
    tint: tintColorLight,
    systemBlue: '#007AFF',
    systemGreen: '#34C759',
    systemIndigo: '#5856D6',
    systemOrange: '#FF9500',
    systemPink: '#FF2D92',
    systemPurple: '#AF52DE',
    systemRed: '#FF3B30',
    systemTeal: '#5AC8FA',
    systemYellow: '#FFCC00',
    
    // Separators
    separator: '#3C3C4349',
    opaqueSeparator: '#C6C6C8',
    
    // Legacy
    text: '#000000',
    textSecondary: '#3C3C43',
    background: '#F2F2F7',
    backgroundElevated: '#FFFFFF',
    icon: '#3C3C43',
    tabIconDefault: '#3C3C43',
    tabIconSelected: tintColorLight,
    border: '#3C3C4349',
    cardBackground: 'rgba(255, 255, 255, 0.95)',
    shadowColor: 'rgba(0, 0, 0, 0.1)',
  },
  dark: {
    // Labels
    label: '#FFFFFF',
    labelSecondary: '#EBEBF5',
    labelTertiary: '#EBEBF599',
    labelQuaternary: '#EBEBF52E',
    
    // Fills
    fill: '#78788066',
    fillSecondary: '#78788052',
    fillTertiary: '#7676803D',
    fillQuaternary: '#74748029',
    
    // Backgrounds
    systemBackground: '#000000',
    secondarySystemBackground: '#1C1C1E',
    tertiarySystemBackground: '#2C2C2E',
    
    // Grouped backgrounds
    systemGroupedBackground: '#000000',
    secondarySystemGroupedBackground: '#1C1C1E',
    tertiarySystemGroupedBackground: '#2C2C2E',
    
    // System colors
    tint: tintColorDark,
    systemBlue: '#0A84FF',
    systemGreen: '#30D158',
    systemIndigo: '#5E5CE6',
    systemOrange: '#FF9F0A',
    systemPink: '#FF2D92',
    systemPurple: '#BF5AF2',
    systemRed: '#FF453A',
    systemTeal: '#64D2FF',
    systemYellow: '#FFD60A',
    
    // Separators
    separator: '#54545899',
    opaqueSeparator: '#38383A',
    
    // Legacy
    text: '#FFFFFF',
    textSecondary: '#EBEBF5',
    background: '#000000',
    backgroundElevated: '#1C1C1E',
    icon: '#EBEBF5',
    tabIconDefault: '#EBEBF5',
    tabIconSelected: tintColorDark,
    border: '#54545899',
    cardBackground: 'rgba(28, 28, 30, 0.95)',
    shadowColor: 'rgba(0, 0, 0, 0.3)',
  },
};

// Apple's Typography Scale (iOS Text Styles)
export const Typography = {
  // Large titles
  largeTitle: {
    fontSize: 34,
    lineHeight: 41,
    fontWeight: '400' as const,
    letterSpacing: 0.37,
  },
  
  // Titles
  title1: {
    fontSize: 28,
    lineHeight: 34,
    fontWeight: '400' as const,
    letterSpacing: 0.36,
  },
  title2: {
    fontSize: 22,
    lineHeight: 28,
    fontWeight: '400' as const,
    letterSpacing: 0.35,
  },
  title3: {
    fontSize: 20,
    lineHeight: 25,
    fontWeight: '400' as const,
    letterSpacing: 0.38,
  },
  
  // Headlines
  headline: {
    fontSize: 17,
    lineHeight: 22,
    fontWeight: '600' as const,
    letterSpacing: -0.41,
  },
  
  // Body text
  body: {
    fontSize: 17,
    lineHeight: 22,
    fontWeight: '400' as const,
    letterSpacing: -0.41,
  },
  
  // Callouts
  callout: {
    fontSize: 16,
    lineHeight: 21,
    fontWeight: '400' as const,
    letterSpacing: -0.32,
  },
  
  // Subheadlines
  subheadline: {
    fontSize: 15,
    lineHeight: 20,
    fontWeight: '400' as const,
    letterSpacing: -0.24,
  },
  
  // Footnotes
  footnote: {
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '400' as const,
    letterSpacing: -0.08,
  },
  
  // Captions
  caption1: {
    fontSize: 12,
    lineHeight: 16,
    fontWeight: '400' as const,
    letterSpacing: 0,
  },
  caption2: {
    fontSize: 11,
    lineHeight: 13,
    fontWeight: '400' as const,
    letterSpacing: 0.07,
  },
};

// Apple's Spacing System (based on 8pt grid)
export const Spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 40,
  xxxl: 48,
};

// Apple's Corner Radius System
export const BorderRadius = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 24,
  continuous: 'continuous' as const, // For iOS continuous corner radius
};

// Apple's Shadow System (iOS-style elevation)
export const Shadows = {
  // Small shadow for cards
  card: {
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3, // Android
  },
  
  // Medium shadow for elevated elements
  elevated: {
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.15,
    shadowRadius: 12,
    elevation: 6, // Android
  },
  
  // Large shadow for modals/overlays
  modal: {
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 8,
    },
    shadowOpacity: 0.25,
    shadowRadius: 16,
    elevation: 12, // Android
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
