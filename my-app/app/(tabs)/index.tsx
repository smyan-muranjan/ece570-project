import React, { useState } from 'react';
import { StyleSheet, ScrollView, View, Text, StatusBar, Pressable, Alert, KeyboardAvoidingView, Platform } from 'react-native';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { Colors, Typography, Spacing, BorderRadius, Shadows } from '@/constants/theme';
import { WeatherInputField } from '@/components/weather-input-field';
import { PredictButton } from '@/components/predict-button';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown, FadeIn } from 'react-native-reanimated';
import { pollenApi, WeatherInput } from '@/services/api';
import { usePrediction } from '@/contexts/prediction-context';
import { router } from 'expo-router';
import DateTimePicker from '@react-native-community/datetimepicker';

export default function HomeScreen() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const { setPrediction, setIsLoading, setError } = usePrediction();
  
  // Form state
  const [date, setDate] = useState(new Date());
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [tempMax, setTempMax] = useState('');
  const [tempMin, setTempMin] = useState('');
  const [precipitation, setPrecipitation] = useState('0');
  const [windSpeed, setWindSpeed] = useState('');
  
  // Advanced optional fields
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [tempAvg, setTempAvg] = useState('');
  const [yesterdayPollen, setYesterdayPollen] = useState('');
  const [threeDaysAgoPollen, setThreeDaysAgoPollen] = useState('');
  const [sevenDaysAgoPollen, setSevenDaysAgoPollen] = useState('');
  
  // Additional historical weather (optional for better accuracy)
  const [avgTempLast30Days, setAvgTempLast30Days] = useState('');
  const [totalRainThisSeason, setTotalRainThisSeason] = useState('');
  const [avgWindLast30Days, setAvgWindLast30Days] = useState('');
  
  const [loading, setLoading] = useState(false);

  const onDateChange = (event: any, selectedDate?: Date) => {
    setShowDatePicker(Platform.OS === 'ios');
    if (selectedDate) {
      setDate(selectedDate);
    }
  };

  const handlePredict = async () => {
    // Validation
    if (!tempMax || !tempMin) {
      Alert.alert('Missing Information', 'Please enter both maximum and minimum temperature.');
      return;
    }

    const max = parseFloat(tempMax);
    const min = parseFloat(tempMin);

    if (isNaN(max) || isNaN(min)) {
      Alert.alert('Invalid Input', 'Please enter valid numbers for temperatures.');
      return;
    }

    if (max < min) {
      Alert.alert('Invalid Range', 'Maximum temperature must be greater than minimum temperature.');
      return;
    }

    setLoading(true);
    setIsLoading(true);
    setError(null);
    
    try {
      const weatherData: WeatherInput = {
        date: date.toISOString().split('T')[0],
        temp_max: max,
        temp_min: min,
        precipitation: parseFloat(precipitation) || 0,
        wind_speed: parseFloat(windSpeed) || undefined,
        temp_avg: tempAvg ? parseFloat(tempAvg) : undefined,
      };
      
      // Build historical pollen array if provided
      const historicalPollen: number[] = [];
      if (yesterdayPollen) historicalPollen.push(parseFloat(yesterdayPollen));
      if (threeDaysAgoPollen && historicalPollen.length > 0) {
        historicalPollen.unshift(parseFloat(threeDaysAgoPollen));
        historicalPollen.unshift(parseFloat(threeDaysAgoPollen)); // Approximate day 2
      }
      if (sevenDaysAgoPollen && historicalPollen.length > 0) {
        // Fill in missing days with interpolation
        while (historicalPollen.length < 7) {
          historicalPollen.unshift(parseFloat(sevenDaysAgoPollen));
        }
      }

      // Build historical weather arrays if provided
      const historicalTemps = avgTempLast30Days ? 
        Array(30).fill(parseFloat(avgTempLast30Days)) : undefined;
      
      const historicalPrecip = totalRainThisSeason ?
        [parseFloat(totalRainThisSeason)] : undefined;
      
      const historicalWind = avgWindLast30Days ?
        Array(30).fill(parseFloat(avgWindLast30Days)) : undefined;

      const prediction = await pollenApi.getDailyPrediction(
        weatherData, 
        historicalPollen.length > 0 ? historicalPollen : undefined,
        historicalTemps,
        historicalPrecip,
        historicalWind
      );
      
      // Also get allergen breakdown with same historical data
      try {
        const allergenData = await pollenApi.identifyAllergens(
          weatherData,
          historicalPollen.length > 0 ? historicalPollen : undefined,
          historicalTemps,
          historicalPrecip,
          historicalWind
        );
        prediction.allergen_breakdown = allergenData.allergens;
        prediction.primary_allergen = allergenData.primary_allergen;
      } catch (allergenError) {
        console.warn('Failed to get allergen breakdown:', allergenError);
        // Continue without allergen data
      }
      
      setPrediction(prediction);
      
      // Navigate to results tab
      router.push('/(tabs)/results');
    } catch (error: any) {
      console.error('Prediction error:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to get prediction';
      setError(errorMessage);
      Alert.alert(
        'Prediction Failed',
        errorMessage + '\n\nMake sure the backend server is running at localhost:8000'
      );
    } finally {
      setLoading(false);
      setIsLoading(false);
    }
  };

  const fillSampleData = () => {
    setTempMax('75');
    setTempMin('55');
    setPrecipitation('0.2');
    setWindSpeed('12');
  };

  const clearForm = () => {
    setTempMax('');
    setTempMin('');
    setPrecipitation('0');
    setWindSpeed('');
    setTempAvg('');
    setYesterdayPollen('');
    setThreeDaysAgoPollen('');
    setSevenDaysAgoPollen('');
    setAvgTempLast30Days('');
    setTotalRainThisSeason('');
    setAvgWindLast30Days('');
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      
        <View 
          style={[
            styles.background, 
            { backgroundColor: isDark ? Colors.dark.systemBackground : Colors.light.systemGroupedBackground }
          ]}
        >
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={styles.keyboardView}
        >
          <ScrollView 
            style={styles.scrollView}
            contentContainerStyle={styles.scrollContent}
            showsVerticalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
          >
            {/* Header */}
            <Animated.View entering={FadeIn} style={styles.header}>
              <View>
                <Text style={[styles.title, { color: isDark ? Colors.dark.label : Colors.light.label }]}>
                  Pollen Predictor
                </Text>
                <Text style={[styles.subtitle, { color: isDark ? Colors.dark.labelSecondary : Colors.light.labelSecondary }]}>
                  Enter your local weather data
                </Text>
              </View>
              <Pressable style={styles.infoButton}>
                <Ionicons 
                  name="information-circle-outline" 
                  size={28} 
                  color={isDark ? Colors.dark.systemBlue : Colors.light.systemBlue} 
                />
              </Pressable>
            </Animated.View>


            {/* Date Section */}
            <Animated.View entering={FadeInDown.delay(200)} style={styles.section}>
                <Text style={[styles.sectionTitle, { color: isDark ? Colors.dark.labelTertiary : Colors.light.labelTertiary }]}>
                  PREDICTION DATE
                </Text>
              <Pressable onPress={() => setShowDatePicker(true)}>
                <View 
                  style={[
                    styles.dateCard, 
                    Shadows.card,
                    { 
                      backgroundColor: isDark ? Colors.dark.secondarySystemGroupedBackground : Colors.light.systemBackground,
                      borderWidth: isDark ? 0 : StyleSheet.hairlineWidth,
                      borderColor: isDark ? 'transparent' : Colors.light.separator,
                    }
                  ]}
                >
                  <View style={styles.dateContent}>
                    <Ionicons name="calendar" size={24} color={isDark ? '#FFF' : '#007AFF'} />
                    <View style={styles.dateTextContainer}>
                    <Text style={[styles.dateText, { color: isDark ? Colors.dark.label : Colors.light.label }]}>
                      {date.toLocaleDateString('en-US', { 
                        weekday: 'long', 
                        year: 'numeric', 
                        month: 'long', 
                        day: 'numeric' 
                      })}
                    </Text>
                    <Text style={[styles.dateHint, { color: isDark ? Colors.dark.labelSecondary : Colors.light.labelSecondary }]}>
                      Tap to change
                    </Text>
                    </View>
                    <Ionicons name="chevron-forward" size={20} color={isDark ? Colors.dark.labelTertiary : Colors.light.labelTertiary} />
                  </View>
                </View>
              </Pressable>
              
              {showDatePicker && (
                <DateTimePicker
                  value={date}
                  mode="date"
                  display={Platform.OS === 'ios' ? 'spinner' : 'default'}
                  onChange={onDateChange}
                  minimumDate={new Date(2020, 0, 1)}
                  maximumDate={new Date(2030, 11, 31)}
                />
              )}
            </Animated.View>

            {/* Weather Inputs */}
            <Animated.View entering={FadeInDown.delay(300)} style={styles.section}>
              <View style={styles.sectionHeader}>
                <Text style={[styles.sectionTitle, { color: isDark ? Colors.dark.labelTertiary : Colors.light.labelTertiary }]}>
                  WEATHER CONDITIONS
                </Text>
                <Pressable 
                  onPress={fillSampleData} 
                  style={[
                    styles.fillButton, 
                    { backgroundColor: isDark ? Colors.dark.systemBlue + '20' : Colors.light.systemBlue + '15' }
                  ]}
                >
                  <Text style={[
                    styles.fillButtonText, 
                    { color: isDark ? Colors.dark.systemBlue : Colors.light.systemBlue }
                  ]}>
                    Fill Sample
                  </Text>
                </Pressable>
              </View>

              <WeatherInputField
                label="Maximum Temperature"
                value={tempMax}
                onChangeText={setTempMax}
                placeholder="75"
                icon="thermometer"
                unit="°F"
              />

              <WeatherInputField
                label="Minimum Temperature"
                value={tempMin}
                onChangeText={setTempMin}
                placeholder="55"
                icon="thermometer-outline"
                unit="°F"
              />

              <WeatherInputField
                label="Precipitation"
                value={precipitation}
                onChangeText={setPrecipitation}
                placeholder="0.0"
                icon="rainy"
                unit="in"
              />

              <WeatherInputField
                label="Average Temperature (Optional)"
                value={tempAvg}
                onChangeText={setTempAvg}
                placeholder="Auto-calculated if blank"
                icon="thermometer"
                unit="°F"
              />

              <WeatherInputField
                label="Wind Speed (Optional)"
                value={windSpeed}
                onChangeText={setWindSpeed}
                placeholder="10"
                icon="flag"
                unit="mph"
              />
            </Animated.View>

            {/* Advanced Options Toggle */}
            <Animated.View entering={FadeInDown.delay(350)} style={styles.advancedToggle}>
              <Pressable 
                onPress={() => setShowAdvanced(!showAdvanced)}
                style={styles.advancedToggleButton}
              >
                <View style={styles.advancedToggleContent}>
                  <Ionicons 
                    name="options" 
                    size={20} 
                    color="#007AFF" 
                    style={styles.advancedToggleIcon}
                  />
                  <Text style={[styles.advancedToggleText, isDark && styles.textDark]}>
                    Advanced Options {showAdvanced ? '(Hide)' : '(Optional - Models work great without these!)'}
                  </Text>
                  <Ionicons 
                    name={showAdvanced ? 'chevron-up' : 'chevron-down'} 
                    size={20} 
                    color={isDark ? '#8E8E93' : '#3C3C43'}
                  />
                </View>
              </Pressable>
            </Animated.View>

            {/* Advanced Fields */}
            {showAdvanced && (
              <Animated.View entering={FadeInDown.delay(100)} style={styles.advancedSection}>
                <View 
                  style={[
                    styles.advancedContainer,
                    Shadows.card,
                    { 
                      backgroundColor: isDark ? Colors.dark.secondarySystemGroupedBackground : Colors.light.systemBackground,
                      borderWidth: isDark ? 0 : StyleSheet.hairlineWidth,
                      borderColor: isDark ? 'transparent' : Colors.light.separator,
                    }
                  ]}
                >
                  <View style={styles.advancedHeader}>
                    <Ionicons name="information-circle" size={18} color="#30D158" />
                    <Text style={[styles.advancedHeaderText, { color: isDark ? Colors.dark.labelSecondary : Colors.light.labelSecondary }]}>
                      Optional data - weather-only models work great without these!
                    </Text>
                  </View>

                  <View style={[styles.divider, { backgroundColor: isDark ? Colors.dark.separator : Colors.light.separator }]} />
                  <Text style={[styles.subsectionTitle, { color: isDark ? Colors.dark.labelTertiary : Colors.light.labelTertiary }]}>
                    HISTORICAL POLLEN (optional - not required for weather-only models)
                  </Text>

                  <WeatherInputField
                    label="Yesterday's Pollen Count"
                    value={yesterdayPollen}
                    onChangeText={setYesterdayPollen}
                    placeholder="e.g., 45"
                    icon="leaf"
                    unit=""
                  />

                  <WeatherInputField
                    label="3 Days Ago Pollen Count"
                    value={threeDaysAgoPollen}
                    onChangeText={setThreeDaysAgoPollen}
                    placeholder="e.g., 52"
                    icon="leaf-outline"
                    unit=""
                  />

                  <WeatherInputField
                    label="7 Days Ago Pollen Count"
                    value={sevenDaysAgoPollen}
                    onChangeText={setSevenDaysAgoPollen}
                    placeholder="e.g., 38"
                    icon="leaf-outline"
                    unit=""
                  />

                  <View style={[styles.divider, { backgroundColor: isDark ? Colors.dark.separator : Colors.light.separator }]} />
                  <Text style={[styles.subsectionTitle, { color: isDark ? Colors.dark.labelTertiary : Colors.light.labelTertiary }]}>
                    HISTORICAL WEATHER (improves biological feature calculations)
                  </Text>

                  <WeatherInputField
                    label="Avg Temperature (Last 30 Days)"
                    value={avgTempLast30Days}
                    onChangeText={setAvgTempLast30Days}
                    placeholder="e.g., 68"
                    icon="thermometer"
                    unit="°F"
                  />

                  <WeatherInputField
                    label="Total Rain This Season"
                    value={totalRainThisSeason}
                    onChangeText={setTotalRainThisSeason}
                    placeholder="e.g., 5.2"
                    icon="rainy"
                    unit="in"
                  />

                  <WeatherInputField
                    label="Avg Wind Speed (Last 30 Days)"
                    value={avgWindLast30Days}
                    onChangeText={setAvgWindLast30Days}
                    placeholder="e.g., 8"
                    icon="flag"
                    unit="mph"
                  />
                </View>
              </Animated.View>
            )}

            {/* Action Buttons */}
            <Animated.View entering={FadeInDown.delay(400)} style={styles.buttonSection}>
              <PredictButton 
                onPress={handlePredict}
                loading={loading}
                disabled={!tempMax || !tempMin}
              />
              
              <Pressable onPress={clearForm} style={styles.clearButton}>
                <Text style={[styles.clearButtonText, { color: isDark ? Colors.dark.systemRed : Colors.light.systemRed }]}>
                  Clear Form
                </Text>
              </Pressable>
            </Animated.View>


            <View style={styles.footer} />
          </ScrollView>
        </KeyboardAvoidingView>
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
  keyboardView: {
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
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: Spacing.lg,
  },
  title: {
    ...Typography.largeTitle,
    fontWeight: '700',
    marginBottom: Spacing.xs,
  },
  subtitle: {
    ...Typography.subheadline,
  },
  subtitleDark: {
    // Will be handled by theme colors
  },
  textDark: {
    // Will be handled by theme colors
  },
  infoButton: {
    padding: Spacing.xs,
  },
  infoCard: {
    borderRadius: BorderRadius.lg,
    overflow: 'hidden',
    marginBottom: Spacing.lg,
  },
  infoContent: {
    flexDirection: 'row',
    padding: Spacing.md,
    gap: Spacing.sm,
    alignItems: 'center',
  },
  infoText: {
    flex: 1,
    ...Typography.callout,
  },
  section: {
    marginBottom: Spacing.lg,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  sectionTitle: {
    ...Typography.caption1,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  fillButton: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.sm,
  },
  fillButtonText: {
    ...Typography.caption1,
    fontWeight: '600',
  },
  dateCard: {
    borderRadius: BorderRadius.lg,
    overflow: 'hidden',
  },
  dateContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.md,
    gap: Spacing.sm,
  },
  dateTextContainer: {
    flex: 1,
  },
  dateText: {
    ...Typography.callout,
    fontWeight: '600',
  },
  dateHint: {
    ...Typography.caption1,
    marginTop: 2,
  },
  buttonSection: {
    marginBottom: Spacing.lg,
  },
  clearButton: {
    alignItems: 'center',
    paddingVertical: Spacing.md,
    marginTop: Spacing.sm,
  },
  clearButtonText: {
    ...Typography.callout,
    fontWeight: '600',
  },
  clearButtonTextDark: {
    // Will be handled by theme colors
  },
  advancedToggle: {
    marginBottom: Spacing.md,
  },
  advancedToggleButton: {
    borderRadius: BorderRadius.md,
    overflow: 'hidden',
  },
  advancedToggleContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.sm,
    gap: Spacing.sm,
  },
  advancedToggleIcon: {
    marginRight: 2,
  },
  advancedToggleText: {
    flex: 1,
    ...Typography.subheadline,
    fontWeight: '500',
  },
  advancedSection: {
    marginBottom: Spacing.lg,
  },
  advancedContainer: {
    borderRadius: BorderRadius.lg,
    overflow: 'hidden',
    padding: Spacing.md,
  },
  advancedHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    marginBottom: Spacing.md,
    paddingHorizontal: Spacing.xs,
  },
  advancedHeaderText: {
    flex: 1,
    ...Typography.caption1,
    fontStyle: 'italic',
  },
  divider: {
    height: StyleSheet.hairlineWidth,
    marginVertical: Spacing.md,
  },
  subsectionTitle: {
    ...Typography.caption2,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: Spacing.sm,
    paddingHorizontal: Spacing.xs,
  },
  footer: {
    height: Spacing.xl,
  },
});
