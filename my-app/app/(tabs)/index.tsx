import React, { useState } from 'react';
import { StyleSheet, ScrollView, View, Text, StatusBar, Pressable, Alert, KeyboardAvoidingView, Platform } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { Colors } from '@/constants/theme';
import { WeatherInputField } from '@/components/weather-input-field';
import { PredictButton } from '@/components/predict-button';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown, FadeIn } from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import { pollenApi, WeatherInput } from '@/services/api';
import { usePrediction } from '@/contexts/prediction-context';
import { router } from 'expo-router';

export default function HomeScreen() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const { setPrediction, setIsLoading, setError } = usePrediction();
  
  // Form state
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [tempMax, setTempMax] = useState('');
  const [tempMin, setTempMin] = useState('');
  const [precipitation, setPrecipitation] = useState('0');
  const [windSpeed, setWindSpeed] = useState('');
  const [loading, setLoading] = useState(false);

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
        date,
        temp_max: max,
        temp_min: min,
        precipitation: parseFloat(precipitation) || 0,
        wind_speed: parseFloat(windSpeed) || 0,
      };

      const prediction = await pollenApi.getDailyPrediction(weatherData);
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
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      
      <LinearGradient
        colors={isDark 
          ? ['#000000', '#1C1C1E', '#1C1C1E'] as const
          : ['#E3F2FD', '#BBDEFB', '#90CAF9'] as const
        }
        style={styles.background}
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
                <Text style={[styles.title, isDark && styles.textDark]}>
                  Pollen Predictor
                </Text>
                <Text style={[styles.subtitle, isDark && styles.subtitleDark]}>
                  Enter your local weather data
                </Text>
              </View>
              <Pressable style={styles.infoButton}>
                <Ionicons name="information-circle-outline" size={28} color={isDark ? '#FFF' : '#007AFF'} />
              </Pressable>
            </Animated.View>

            {/* Info Card */}
            <Animated.View entering={FadeInDown.delay(100)}>
              <BlurView 
                intensity={isDark ? 20 : 80} 
                tint={isDark ? 'dark' : 'light'}
                style={styles.infoCard}
              >
                <View style={styles.infoContent}>
                  <Ionicons name="bulb-outline" size={24} color="#FFD60A" />
                  <Text style={[styles.infoText, isDark && styles.textDark]}>
                    No pollen measurements in your area? Use our ML model to predict pollen levels from weather data alone!
                  </Text>
                </View>
              </BlurView>
            </Animated.View>

            {/* Date Section */}
            <Animated.View entering={FadeInDown.delay(200)} style={styles.section}>
              <Text style={[styles.sectionTitle, isDark && styles.subtitleDark]}>
                PREDICTION DATE
              </Text>
              <BlurView 
                intensity={isDark ? 20 : 80} 
                tint={isDark ? 'dark' : 'light'}
                style={styles.dateCard}
              >
                <View style={styles.dateContent}>
                  <Ionicons name="calendar" size={24} color={isDark ? '#FFF' : '#007AFF'} />
                  <Text style={[styles.dateText, isDark && styles.textDark]}>
                    {new Date(date).toLocaleDateString('en-US', { 
                      weekday: 'long', 
                      year: 'numeric', 
                      month: 'long', 
                      day: 'numeric' 
                    })}
                  </Text>
                </View>
              </BlurView>
            </Animated.View>

            {/* Weather Inputs */}
            <Animated.View entering={FadeInDown.delay(300)} style={styles.section}>
              <View style={styles.sectionHeader}>
                <Text style={[styles.sectionTitle, isDark && styles.subtitleDark]}>
                  WEATHER CONDITIONS
                </Text>
                <Pressable onPress={fillSampleData} style={styles.fillButton}>
                  <Text style={styles.fillButtonText}>Fill Sample</Text>
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
                label="Wind Speed (Optional)"
                value={windSpeed}
                onChangeText={setWindSpeed}
                placeholder="10"
                icon="flag"
                unit="mph"
              />
            </Animated.View>

            {/* Action Buttons */}
            <Animated.View entering={FadeInDown.delay(400)} style={styles.buttonSection}>
              <PredictButton 
                onPress={handlePredict}
                loading={loading}
                disabled={!tempMax || !tempMin}
              />
              
              <Pressable onPress={clearForm} style={styles.clearButton}>
                <Text style={[styles.clearButtonText, isDark && styles.clearButtonTextDark]}>
                  Clear Form
                </Text>
              </Pressable>
            </Animated.View>

            {/* Features Info */}
            <Animated.View entering={FadeInDown.delay(500)} style={styles.featuresSection}>
              <Text style={[styles.sectionTitle, isDark && styles.subtitleDark]}>
                WHAT YOU'LL GET
              </Text>
              
              {[
                { icon: 'analytics', title: 'Pollen Severity Score', desc: '0-10 scale prediction' },
                { icon: 'leaf', title: 'Allergen Breakdown', desc: 'Tree, Grass, Ragweed, Weed levels' },
                { icon: 'shield-checkmark', title: 'Health Recommendations', desc: 'Personalized advice based on severity' },
              ].map((feature, index) => (
                <BlurView 
                  key={feature.title}
                  intensity={isDark ? 20 : 80} 
                  tint={isDark ? 'dark' : 'light'}
                  style={styles.featureCard}
                >
                  <View style={styles.featureContent}>
                    <View style={[styles.featureIcon, { backgroundColor: isDark ? '#007AFF20' : '#007AFF15' }]}>
                      <Ionicons name={feature.icon as any} size={24} color="#007AFF" />
                    </View>
                    <View style={styles.featureText}>
                      <Text style={[styles.featureTitle, isDark && styles.textDark]}>
                        {feature.title}
                      </Text>
                      <Text style={[styles.featureDesc, isDark && styles.subtitleDark]}>
                        {feature.desc}
                      </Text>
                    </View>
                  </View>
                </BlurView>
              ))}
            </Animated.View>

            <View style={styles.footer} />
          </ScrollView>
        </KeyboardAvoidingView>
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
  keyboardView: {
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
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 20,
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
  infoButton: {
    padding: 4,
  },
  infoCard: {
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    marginBottom: 24,
  },
  infoContent: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    alignItems: 'center',
  },
  infoText: {
    flex: 1,
    fontSize: 14,
    lineHeight: 20,
    color: '#000',
  },
  section: {
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#8E8E93',
    letterSpacing: 0.5,
  },
  fillButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    backgroundColor: 'rgba(0, 122, 255, 0.15)',
  },
  fillButtonText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#007AFF',
  },
  dateCard: {
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  dateContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    gap: 12,
  },
  dateText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#000',
  },
  buttonSection: {
    marginBottom: 24,
  },
  clearButton: {
    alignItems: 'center',
    paddingVertical: 16,
    marginTop: 12,
  },
  clearButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FF453A',
  },
  clearButtonTextDark: {
    color: '#FF6961',
  },
  featuresSection: {
    marginBottom: 24,
  },
  featureCard: {
    borderRadius: 12,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    marginBottom: 8,
  },
  featureContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    gap: 12,
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
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
    marginBottom: 2,
  },
  featureDesc: {
    fontSize: 14,
    color: '#8E8E93',
  },
  footer: {
    height: 40,
  },
});
