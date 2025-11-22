import React from 'react';
import { StyleSheet, ScrollView, View, Text, StatusBar, Pressable } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { PollenColors, AllergenColors } from '@/constants/theme';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown, FadeIn } from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import { usePrediction } from '@/contexts/prediction-context';

export default function ResultsScreen() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const { prediction } = usePrediction();

  const getSeverityColor = (severity: number) => {
    if (severity <= 2) return PollenColors.veryLow;
    if (severity <= 4) return PollenColors.low;
    if (severity <= 6) return PollenColors.moderate;
    if (severity <= 8) return PollenColors.high;
    return PollenColors.severe;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  const getAllergenColor = (type: string) => {
    const typeLower = type.toLowerCase();
    if (typeLower === 'tree') return AllergenColors.tree;
    if (typeLower === 'grass') return AllergenColors.grass;
    if (typeLower === 'ragweed') return AllergenColors.ragweed;
    if (typeLower === 'weed') return AllergenColors.weed;
    return '#8E8E93';
  };

  if (!prediction) {
    return (
      <View style={styles.container}>
        <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
        <LinearGradient
          colors={isDark 
            ? ['#000000', '#1C1C1E'] as const
            : ['#F2F2F7', '#E5E5EA'] as const
          }
          style={styles.background}
        >
          <View style={styles.emptyState}>
            <Ionicons name="analytics-outline" size={80} color="#8E8E93" />
            <Text style={[styles.emptyTitle, isDark && styles.textDark]}>
              No Predictions Yet
            </Text>
            <Text style={[styles.emptySubtitle, isDark && styles.subtitleDark]}>
              Make your first prediction on the Predict tab
            </Text>
          </View>
        </LinearGradient>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      
      <LinearGradient
        colors={isDark 
          ? ['#000000', '#1C1C1E', '#1C1C1E'] as const
          : ['#E3F2FD', '#BBDEFB', '#F2F2F7'] as const
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
              Prediction Results
            </Text>
            <Text style={[styles.subtitle, isDark && styles.subtitleDark]}>
              {formatDate(prediction.prediction.date)}
            </Text>
          </Animated.View>

          {/* Main Severity Card */}
          <Animated.View entering={FadeInDown.delay(100)}>
            <LinearGradient
              colors={[
                getSeverityColor(prediction.prediction.severity_score) + '40',
                getSeverityColor(prediction.prediction.severity_score) + '20',
              ] as const}
              style={styles.severityCard}
            >
              <View style={styles.severityContent}>
                <View style={styles.severityTop}>
                  <View>
                    <Text style={[styles.severityLabel, isDark && styles.subtitleDark]}>
                      Pollen Severity
                    </Text>
                    <Text style={[styles.severityValue, { color: getSeverityColor(prediction.prediction.severity_score) }]}>
                      {prediction.prediction.severity_score.toFixed(1)}
                    </Text>
                  </View>
                  <View style={[styles.severityBadge, { backgroundColor: getSeverityColor(prediction.prediction.severity_score) }]}>
                    <Text style={styles.severityBadgeText}>{prediction.prediction.severity_level}</Text>
                  </View>
                </View>

                <View style={styles.confidenceContainer}>
                  <Ionicons name="rocket" size={16} color={getSeverityColor(prediction.prediction.severity_score)} />
                  <Text style={[styles.confidenceText, isDark && styles.textDark]}>
                    {((prediction.prediction.confidence || 0) * 100).toFixed(0)}% Confidence • Weather-Only Model
                  </Text>
                </View>
              </View>
            </LinearGradient>
          </Animated.View>

          {/* Model Info */}
          <Animated.View entering={FadeInDown.delay(150)}>
            <BlurView 
              intensity={isDark ? 20 : 80} 
              tint={isDark ? 'dark' : 'light'}
              style={styles.modelInfoCard}
            >
              <View style={styles.modelInfoContent}>
                <View style={styles.modelInfoIcon}>
                  <Ionicons name="flash" size={20} color="#30D158" />
                </View>
                <View style={styles.modelInfoText}>
                  <Text style={[styles.modelInfoTitle, isDark && styles.textDark]}>
                    Advanced Weather-Only Model
                  </Text>
                  <Text style={[styles.modelInfoDesc, isDark && styles.subtitleDark]}>
                    Uses VPD, Ventilation Index, and biological features • 47.9% better accuracy
                  </Text>
                </View>
              </View>
            </BlurView>
          </Animated.View>

          {/* Recommendation */}
          <Animated.View entering={FadeInDown.delay(200)}>
            <BlurView 
              intensity={isDark ? 20 : 80} 
              tint={isDark ? 'dark' : 'light'}
              style={styles.recommendationCard}
            >
              <View style={styles.recommendationContent}>
                <View style={[styles.iconCircle, { backgroundColor: getSeverityColor(prediction.prediction.severity_score) + '20' }]}>
                  <Ionicons name="warning" size={24} color={getSeverityColor(prediction.prediction.severity_score)} />
                </View>
                <View style={styles.recommendationText}>
                  <Text style={[styles.recommendationTitle, isDark && styles.textDark]}>
                    Health Advisory
                  </Text>
                  <Text style={[styles.recommendationDesc, isDark && styles.subtitleDark]}>
                    {prediction.recommendation}
                  </Text>
                </View>
              </View>
            </BlurView>
          </Animated.View>

          {/* Allergen Breakdown */}
          {prediction.allergen_breakdown && prediction.allergen_breakdown.length > 0 && (
            <Animated.View entering={FadeInDown.delay(300)} style={styles.section}>
              <Text style={[styles.sectionTitle, isDark && styles.subtitleDark]}>
                ALLERGEN BREAKDOWN
              </Text>

              {/* Percentage Bar */}
              <View style={styles.percentageBar}>
                {prediction.allergen_breakdown.map((allergen, index) => (
                  <View 
                    key={allergen.allergen_type}
                    style={[
                      styles.percentageSegment,
                      { 
                        width: `${allergen.contribution_pct}%`,
                        backgroundColor: getAllergenColor(allergen.allergen_type),
                        borderTopLeftRadius: index === 0 ? 6 : 0,
                        borderBottomLeftRadius: index === 0 ? 6 : 0,
                        borderTopRightRadius: index === prediction.allergen_breakdown!.length - 1 ? 6 : 0,
                        borderBottomRightRadius: index === prediction.allergen_breakdown!.length - 1 ? 6 : 0,
                      }
                    ]}
                  />
                ))}
              </View>

              {/* Allergen Cards */}
              {prediction.allergen_breakdown.map((allergen, index) => (
                <BlurView 
                  key={allergen.allergen_type}
                  intensity={isDark ? 20 : 80} 
                  tint={isDark ? 'dark' : 'light'}
                  style={styles.allergenCard}
                >
                  <View style={styles.allergenContent}>
                    <View style={[styles.allergenIcon, { backgroundColor: getAllergenColor(allergen.allergen_type) + '20' }]}>
                      <Ionicons name="leaf" size={24} color={getAllergenColor(allergen.allergen_type)} />
                    </View>
                    
                    <View style={styles.allergenDetails}>
                      <Text style={[styles.allergenName, isDark && styles.textDark]}>
                        {allergen.allergen_type}
                      </Text>
                      <View style={styles.allergenStats}>
                        <Text style={[styles.allergenPercentage, { color: getAllergenColor(allergen.allergen_type) }]}>
                          {allergen.contribution_pct.toFixed(1)}%
                        </Text>
                        <Text style={[styles.allergenSeparator, isDark && styles.subtitleDark]}>
                          •
                        </Text>
                        <Text style={[styles.allergenSeverity, isDark && styles.subtitleDark]}>
                          Severity: {allergen.severity_score.toFixed(1)}
                        </Text>
                      </View>
                    </View>

                    <View style={[styles.allergenProgress, { width: 60 }]}>
                      <View style={styles.allergenProgressBg}>
                        <View 
                          style={[
                            styles.allergenProgressFill,
                            { 
                              width: `${(allergen.severity_score / 10) * 100}%`,
                              backgroundColor: getAllergenColor(allergen.allergen_type) 
                            }
                          ]}
                        />
                      </View>
                    </View>
                  </View>
                </BlurView>
              ))}
            </Animated.View>
          )}

          {/* Action Button */}
          <Animated.View entering={FadeInDown.delay(400)} style={styles.actionSection}>
            <Pressable style={styles.newPredictionButton}>
              <LinearGradient
                colors={['#007AFF', '#0051D5'] as const}
                style={styles.buttonGradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
              >
                <Ionicons name="add-circle-outline" size={24} color="#FFF" />
                <Text style={styles.buttonText}>Make New Prediction</Text>
              </LinearGradient>
            </Pressable>
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
  severityCard: {
    borderRadius: 24,
    padding: 24,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.15,
    shadowRadius: 16,
    elevation: 8,
  },
  severityContent: {
    gap: 16,
  },
  severityTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  severityLabel: {
    fontSize: 14,
    color: '#8E8E93',
    marginBottom: 8,
    fontWeight: '500',
  },
  severityValue: {
    fontSize: 56,
    fontWeight: '200',
    letterSpacing: -2,
  },
  severityBadge: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  severityBadgeText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#FFF',
  },
  confidenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  confidenceText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#000',
  },
  modelInfoCard: {
    borderRadius: 12,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(48, 209, 88, 0.3)',
    marginBottom: 16,
  },
  modelInfoContent: {
    flexDirection: 'row',
    padding: 12,
    gap: 10,
    alignItems: 'center',
  },
  modelInfoIcon: {
    width: 32,
    height: 32,
    borderRadius: 8,
    backgroundColor: 'rgba(48, 209, 88, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modelInfoText: {
    flex: 1,
  },
  modelInfoTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#000',
    marginBottom: 2,
  },
  modelInfoDesc: {
    fontSize: 12,
    color: '#8E8E93',
    lineHeight: 16,
  },
  recommendationCard: {
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    marginBottom: 24,
  },
  recommendationContent: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
  },
  iconCircle: {
    width: 48,
    height: 48,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  recommendationText: {
    flex: 1,
  },
  recommendationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
    marginBottom: 4,
  },
  recommendationDesc: {
    fontSize: 14,
    lineHeight: 20,
    color: '#8E8E93',
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
  percentageBar: {
    flexDirection: 'row',
    height: 12,
    borderRadius: 6,
    overflow: 'hidden',
    marginBottom: 16,
  },
  percentageSegment: {
    height: '100%',
  },
  allergenCard: {
    borderRadius: 12,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    marginBottom: 8,
  },
  allergenContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    gap: 12,
  },
  allergenIcon: {
    width: 48,
    height: 48,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  allergenDetails: {
    flex: 1,
  },
  allergenName: {
    fontSize: 17,
    fontWeight: '600',
    color: '#000',
    marginBottom: 4,
  },
  allergenStats: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  allergenPercentage: {
    fontSize: 14,
    fontWeight: '600',
  },
  allergenSeparator: {
    fontSize: 14,
    color: '#8E8E93',
  },
  allergenSeverity: {
    fontSize: 13,
    color: '#8E8E93',
  },
  allergenProgress: {
    height: 40,
    justifyContent: 'center',
  },
  allergenProgressBg: {
    height: 6,
    backgroundColor: 'rgba(142, 142, 147, 0.2)',
    borderRadius: 3,
    overflow: 'hidden',
  },
  allergenProgressFill: {
    height: '100%',
    borderRadius: 3,
  },
  weatherCard: {
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  weatherGrid: {
    flexDirection: 'row',
    padding: 16,
  },
  weatherItem: {
    flex: 1,
    alignItems: 'center',
    gap: 6,
  },
  weatherLabel: {
    fontSize: 12,
    color: '#8E8E93',
  },
  weatherValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
  },
  actionSection: {
    marginBottom: 24,
  },
  newPredictionButton: {
    borderRadius: 16,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 12,
    elevation: 5,
  },
  buttonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    gap: 8,
  },
  buttonText: {
    fontSize: 17,
    fontWeight: '600',
    color: '#FFF',
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 40,
  },
  emptyTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: '#000',
    marginTop: 20,
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 16,
    color: '#8E8E93',
    textAlign: 'center',
  },
  footer: {
    height: 40,
  },
});
