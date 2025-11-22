import React from 'react';
import { StyleSheet, ScrollView, View, Text, StatusBar, Pressable } from 'react-native';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { PollenColors, AllergenColors, Colors, Typography, Spacing, BorderRadius, Shadows } from '@/constants/theme';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown, FadeIn } from 'react-native-reanimated';
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
        <View 
          style={[
            styles.background, 
            { backgroundColor: isDark ? Colors.dark.systemBackground : Colors.light.systemGroupedBackground }
          ]}
        >
          <View style={styles.emptyState}>
            <Ionicons name="analytics-outline" size={80} color="#8E8E93" />
            <Text style={[styles.emptyTitle, { color: isDark ? Colors.dark.label : Colors.light.label }]}>
              No Predictions Yet
            </Text>
            <Text style={[styles.emptySubtitle, { color: isDark ? Colors.dark.labelSecondary : Colors.light.labelSecondary }]}>
              Make your first prediction on the Predict tab
            </Text>
          </View>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      
      <View 
        style={[
          styles.background, 
          { backgroundColor: isDark ? Colors.dark.systemBackground : Colors.light.systemGroupedBackground }
        ]}
      >
        <ScrollView 
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Header */}
          <Animated.View entering={FadeIn} style={styles.header}>
            <Text style={[styles.title, { color: isDark ? Colors.dark.label : Colors.light.label }]}>
              Prediction Results
            </Text>
            <Text style={[styles.subtitle, { color: isDark ? Colors.dark.labelSecondary : Colors.light.labelSecondary }]}>
              {formatDate(prediction.prediction.date)}
            </Text>
          </Animated.View>

          {/* Main Severity Card */}
          <Animated.View entering={FadeInDown.delay(100)}>
            <View 
              style={[
                styles.severityCard,
                Shadows.elevated,
                { 
                  backgroundColor: isDark ? Colors.dark.secondarySystemGroupedBackground : Colors.light.systemBackground,
                  borderWidth: isDark ? 0 : StyleSheet.hairlineWidth,
                  borderColor: isDark ? 'transparent' : Colors.light.separator,
                }
              ]}
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
                  <Ionicons name="shield-checkmark" size={16} color={getSeverityColor(prediction.prediction.severity_score)} />
                  <Text style={[styles.confidenceText, { color: isDark ? Colors.dark.label : Colors.light.label }]}>
                    {((prediction.prediction.confidence || 0) * 100).toFixed(0)}% Model Confidence
                  </Text>
                </View>
              </View>
            </View>
          </Animated.View>


          {/* Recommendation */}
          <Animated.View entering={FadeInDown.delay(200)}>
            <View 
              style={[
                styles.recommendationCard,
                Shadows.card,
                { 
                  backgroundColor: isDark ? Colors.dark.secondarySystemGroupedBackground : Colors.light.systemBackground,
                  borderWidth: isDark ? 0 : StyleSheet.hairlineWidth,
                  borderColor: isDark ? 'transparent' : Colors.light.separator,
                }
              ]}
            >
              <View style={styles.recommendationContent}>
                <View style={[styles.iconCircle, { backgroundColor: getSeverityColor(prediction.prediction.severity_score) + '20' }]}>
                  <Ionicons name="warning" size={24} color={getSeverityColor(prediction.prediction.severity_score)} />
                </View>
                <View style={styles.recommendationText}>
                  <Text style={[styles.recommendationTitle, { color: isDark ? Colors.dark.label : Colors.light.label }]}>
                    Health Advisory
                  </Text>
                  <Text style={[styles.recommendationDesc, { color: isDark ? Colors.dark.labelSecondary : Colors.light.labelSecondary }]}>
                    {prediction.recommendation}
                  </Text>
                </View>
              </View>
            </View>
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
                <View 
                  key={allergen.allergen_type}
                  style={[
                    styles.allergenCard,
                    Shadows.card,
                    { 
                      backgroundColor: isDark ? Colors.dark.secondarySystemGroupedBackground : Colors.light.systemBackground,
                      borderWidth: isDark ? 0 : StyleSheet.hairlineWidth,
                      borderColor: isDark ? 'transparent' : Colors.light.separator,
                    }
                  ]}
                >
                  <View style={styles.allergenContent}>
                    <View style={[styles.allergenIcon, { backgroundColor: getAllergenColor(allergen.allergen_type) + '20' }]}>
                      <Ionicons name="leaf" size={24} color={getAllergenColor(allergen.allergen_type)} />
                    </View>
                    
                    <View style={styles.allergenDetails}>
                      <Text style={[styles.allergenName, { color: isDark ? Colors.dark.label : Colors.light.label }]}>
                        {allergen.allergen_type}
                      </Text>
                      <View style={styles.allergenStats}>
                        <Text style={[styles.allergenPercentage, { color: getAllergenColor(allergen.allergen_type) }]}>
                          {allergen.contribution_pct.toFixed(1)}%
                        </Text>
                        <Text style={[styles.allergenSeparator, { color: isDark ? Colors.dark.labelSecondary : Colors.light.labelSecondary }]}>
                          •
                        </Text>
                        <Text style={[styles.allergenSeverity, { color: isDark ? Colors.dark.labelSecondary : Colors.light.labelSecondary }]}>
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
                </View>
              ))}
            </Animated.View>
          )}

          {/* Action Button */}
          <Animated.View entering={FadeInDown.delay(400)} style={styles.actionSection}>
            <Pressable style={styles.newPredictionButton}>
              <View 
                style={[
                  styles.buttonGradient,
                  Shadows.card,
                  { backgroundColor: isDark ? Colors.dark.systemBlue : Colors.light.systemBlue }
                ]}
              >
                <Ionicons name="add-circle-outline" size={24} color="#FFF" />
                <Text style={styles.buttonText}>Make New Prediction</Text>
              </View>
            </Pressable>
          </Animated.View>

          <View style={styles.footer} />
        </ScrollView>
      </View>
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
    paddingTop: Spacing.xxxl + Spacing.md,
    paddingHorizontal: Spacing.md,
  },
  header: {
    marginBottom: Spacing.lg,
  },
  title: {
    ...Typography.largeTitle,
    fontWeight: '700',
    marginBottom: Spacing.xs,
  },
  subtitle: {
    ...Typography.callout,
  },
  subtitleDark: {
    // Will be handled by theme colors
  },
  textDark: {
    // Will be handled by theme colors
  },
  severityCard: {
    borderRadius: BorderRadius.xl,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
  },
  severityContent: {
    gap: Spacing.md,
  },
  severityTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  severityLabel: {
    ...Typography.subheadline,
    marginBottom: Spacing.xs,
    fontWeight: '500',
  },
  severityValue: {
    fontSize: 56,
    fontWeight: '200',
    letterSpacing: -2,
  },
  severityBadge: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.xl,
  },
  severityBadgeText: {
    ...Typography.subheadline,
    fontWeight: '600',
    color: '#FFF',
  },
  confidenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  confidenceText: {
    ...Typography.subheadline,
    fontWeight: '500',
  },
  recommendationCard: {
    borderRadius: BorderRadius.lg,
    overflow: 'hidden',
    marginBottom: Spacing.lg,
  },
  recommendationContent: {
    flexDirection: 'row',
    padding: Spacing.md,
    gap: Spacing.sm,
  },
  iconCircle: {
    width: 48,
    height: 48,
    borderRadius: BorderRadius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  recommendationText: {
    flex: 1,
  },
  recommendationTitle: {
    ...Typography.callout,
    fontWeight: '600',
    marginBottom: Spacing.xs,
  },
  recommendationDesc: {
    ...Typography.subheadline,
    lineHeight: 20,
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
    paddingHorizontal: Spacing.xl,
  },
  emptyTitle: {
    ...Typography.title2,
    fontWeight: '600',
    marginTop: Spacing.md,
    marginBottom: Spacing.xs,
  },
  emptySubtitle: {
    ...Typography.callout,
    textAlign: 'center',
  },
  footer: {
    height: Spacing.xl,
  },
});
