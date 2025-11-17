import React from 'react';
import { StyleSheet, ScrollView, View, Text, StatusBar, Linking } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown, FadeIn } from 'react-native-reanimated';
import { BlurView } from 'expo-blur';

export default function TabTwoScreen() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const features = [
    {
      icon: 'analytics',
      title: 'ML-Powered Predictions',
      description: 'Uses Random Forest and XGBoost models trained on historical pollen and weather data.',
      color: '#007AFF',
    },
    {
      icon: 'leaf',
      title: 'Allergen Breakdown',
      description: 'Get detailed predictions for Tree, Grass, Ragweed, and Weed pollen levels.',
      color: '#34C759',
    },
    {
      icon: 'location',
      title: 'Works Anywhere',
      description: 'Perfect for areas without pollen count monitoring stations. Only weather data needed.',
      color: '#FF9500',
    },
    {
      icon: 'shield-checkmark',
      title: 'High Accuracy',
      description: 'Model achieves 86% R² score with average error less than 1 severity level.',
      color: '#BF5AF2',
    },
  ];

  const severityLevels = [
    { level: '0-2', label: 'Low', color: '#34C759', description: 'Enjoy outdoor activities' },
    { level: '3-4', label: 'Moderate', color: '#FFD60A', description: 'Monitor symptoms' },
    { level: '5-6', label: 'High', color: '#FF9500', description: 'Limit outdoor exposure' },
    { level: '7-8', label: 'Very High', color: '#FF453A', description: 'Stay indoors if possible' },
    { level: '9-10', label: 'Severe', color: '#BF5AF2', description: 'Avoid outdoor activities' },
  ];

  return (
    <View style={styles.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      
      <LinearGradient
        colors={isDark 
          ? ['#000000', '#1C1C1E', '#1C1C1E'] as const
          : ['#F2F2F7', '#E5E5EA', '#F2F2F7'] as const
        }
        style={styles.background}
      >
        <ScrollView 
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Header */}
          <Animated.View entering={FadeIn} style={styles.header}>
            <Text style={[styles.title, isDark && styles.textDark]}>
              About Pollen Predictor
            </Text>
            <Text style={[styles.subtitle, isDark && styles.subtitleDark]}>
              Science-backed pollen forecasting
            </Text>
          </Animated.View>

          {/* Hero Card */}
          <Animated.View entering={FadeInDown.delay(100)}>
            <LinearGradient
              colors={['#007AFF', '#0051D5'] as const}
              style={styles.heroCard}
            >
              <View style={styles.heroContent}>
                <Ionicons name="analytics" size={48} color="#FFF" />
                <Text style={styles.heroTitle}>
                  Predict Pollen Levels
                </Text>
                <Text style={styles.heroSubtitle}>
                  Using only weather data and machine learning
                </Text>
              </View>
            </LinearGradient>
          </Animated.View>

          {/* Features */}
          <Animated.View entering={FadeInDown.delay(200)} style={styles.section}>
            <Text style={[styles.sectionTitle, isDark && styles.subtitleDark]}>
              KEY FEATURES
            </Text>
            
            {features.map((feature, index) => (
              <BlurView 
                key={feature.title}
                intensity={isDark ? 20 : 80} 
                tint={isDark ? 'dark' : 'light'}
                style={styles.featureCard}
              >
                <View style={styles.featureContent}>
                  <View style={[styles.featureIcon, { backgroundColor: feature.color + '20' }]}>
                    <Ionicons name={feature.icon as any} size={28} color={feature.color} />
                  </View>
                  <View style={styles.featureText}>
                    <Text style={[styles.featureTitle, isDark && styles.textDark]}>
                      {feature.title}
                    </Text>
                    <Text style={[styles.featureDesc, isDark && styles.subtitleDark]}>
                      {feature.description}
                    </Text>
                  </View>
                </View>
              </BlurView>
            ))}
          </Animated.View>

          {/* Severity Scale */}
          <Animated.View entering={FadeInDown.delay(300)} style={styles.section}>
            <Text style={[styles.sectionTitle, isDark && styles.subtitleDark]}>
              SEVERITY SCALE (0-10)
            </Text>
            
            {severityLevels.map((item, index) => (
              <BlurView 
                key={item.label}
                intensity={isDark ? 20 : 80} 
                tint={isDark ? 'dark' : 'light'}
                style={styles.severityCard}
              >
                <View style={styles.severityContent}>
                  <View style={[styles.severityIndicator, { backgroundColor: item.color }]} />
                  <View style={styles.severityInfo}>
                    <View style={styles.severityTop}>
                      <Text style={[styles.severityLabel, isDark && styles.textDark]}>
                        {item.label}
                      </Text>
                      <Text style={[styles.severityRange, { color: item.color }]}>
                        {item.level}
                      </Text>
                    </View>
                    <Text style={[styles.severityDesc, isDark && styles.subtitleDark]}>
                      {item.description}
                    </Text>
                  </View>
                </View>
              </BlurView>
            ))}
          </Animated.View>

          {/* How It Works */}
          <Animated.View entering={FadeInDown.delay(400)} style={styles.section}>
            <Text style={[styles.sectionTitle, isDark && styles.subtitleDark]}>
              HOW IT WORKS
            </Text>
            
            <BlurView 
              intensity={isDark ? 20 : 80} 
              tint={isDark ? 'dark' : 'light'}
              style={styles.infoCard}
            >
              <View style={styles.infoContent}>
                <View style={styles.step}>
                  <View style={styles.stepNumber}>
                    <Text style={styles.stepNumberText}>1</Text>
                  </View>
                  <Text style={[styles.stepText, isDark && styles.textDark]}>
                    Enter your local weather data (temperature, precipitation, wind)
                  </Text>
                </View>
                
                <View style={styles.stepDivider} />
                
                <View style={styles.step}>
                  <View style={styles.stepNumber}>
                    <Text style={styles.stepNumberText}>2</Text>
                  </View>
                  <Text style={[styles.stepText, isDark && styles.textDark]}>
                    Our ML model analyzes weather patterns and seasonal trends
                  </Text>
                </View>
                
                <View style={styles.stepDivider} />
                
                <View style={styles.step}>
                  <View style={styles.stepNumber}>
                    <Text style={styles.stepNumberText}>3</Text>
                  </View>
                  <Text style={[styles.stepText, isDark && styles.textDark]}>
                    Get instant pollen severity predictions with allergen breakdown
                  </Text>
                </View>
              </View>
            </BlurView>
          </Animated.View>

          {/* Model Info */}
          <Animated.View entering={FadeInDown.delay(500)} style={styles.section}>
            <Text style={[styles.sectionTitle, isDark && styles.subtitleDark]}>
              MODEL PERFORMANCE
            </Text>
            
            <BlurView 
              intensity={isDark ? 20 : 80} 
              tint={isDark ? 'dark' : 'light'}
              style={styles.statsCard}
            >
              <View style={styles.statsGrid}>
                <View style={styles.statItem}>
                  <Text style={[styles.statValue, isDark && styles.textDark]}>0.72</Text>
                  <Text style={[styles.statLabel, isDark && styles.subtitleDark]}>MAE</Text>
                  <Text style={[styles.statDesc, isDark && styles.subtitleDark]}>Mean Error</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={[styles.statValue, isDark && styles.textDark]}>0.86</Text>
                  <Text style={[styles.statLabel, isDark && styles.subtitleDark]}>R²</Text>
                  <Text style={[styles.statDesc, isDark && styles.subtitleDark]}>Accuracy</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={[styles.statValue, isDark && styles.textDark]}>200</Text>
                  <Text style={[styles.statLabel, isDark && styles.subtitleDark]}>Trees</Text>
                  <Text style={[styles.statDesc, isDark && styles.subtitleDark]}>Random Forest</Text>
                </View>
              </View>
            </BlurView>
          </Animated.View>

          <View style={styles.footer} />
        </ScrollView>
      </LinearGradient>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  background: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingTop: 60,
    paddingHorizontal: 20,
  },
  header: {
    marginBottom: 24,
  },
  title: {
    fontSize: 34,
    fontWeight: '700',
    color: '#000',
    marginBottom: 4,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 16,
    color: '#8E8E93',
    fontWeight: '400',
  },
  subtitleDark: {
    color: '#8E8E93',
  },
  textDark: {
    color: '#FFF',
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#8E8E93',
    marginBottom: 12,
    letterSpacing: 0.5,
  },
  heroCard: {
    borderRadius: 24,
    overflow: 'hidden',
    marginBottom: 24,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  heroContent: {
    padding: 32,
    alignItems: 'center',
  },
  heroTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: '#000',
    marginBottom: 8,
    textAlign: 'center',
  },
  heroSubtitle: {
    fontSize: 16,
    color: '#8E8E93',
    textAlign: 'center',
    lineHeight: 22,
  },
  featureCard: {
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  featureContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    gap: 16,
  },
  featureIcon: {
    width: 48,
    height: 48,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  featureText: {
    flex: 1,
  },
  featureTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: '#000',
    marginBottom: 4,
  },
  featureDesc: {
    fontSize: 14,
    color: '#8E8E93',
    lineHeight: 20,
  },
  severityContent: {
    gap: 8,
  },
  severityCard: {
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  severityIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    gap: 12,
  },
  severityInfo: {
    flex: 1,
  },
  severityTop: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 8,
    marginBottom: 2,
  },
  severityLabel: {
    fontSize: 17,
    fontWeight: '600',
    color: '#FFF',
  },
  severityRange: {
    fontSize: 13,
    color: 'rgba(255, 255, 255, 0.7)',
  },
  severityDesc: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  infoCard: {
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 24,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  infoContent: {
    padding: 24,
  },
  step: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  stepNumber: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#007AFF',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  stepNumberText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#FFF',
  },
  stepText: {
    flex: 1,
    fontSize: 15,
    color: '#000',
    lineHeight: 22,
  },
  stepDivider: {
    height: 1,
    backgroundColor: 'rgba(142, 142, 147, 0.2)',
    marginBottom: 16,
  },
  statsCard: {
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  statsGrid: {
    flexDirection: 'row',
    padding: 24,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 32,
    fontWeight: '700',
    color: '#000',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 15,
    fontWeight: '600',
    color: '#8E8E93',
    marginBottom: 2,
  },
  statDesc: {
    fontSize: 13,
    color: '#8E8E93',
  },
  statDivider: {
    width: 1,
    backgroundColor: 'rgba(142, 142, 147, 0.2)',
    marginHorizontal: 16,
  },
  footer: {
    height: 40,
  },
});
