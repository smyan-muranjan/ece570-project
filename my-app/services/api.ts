import axios from 'axios';

// Update this to your backend URL when running on device/emulator
// For iOS simulator: http://localhost:8000
// For Android emulator: http://10.0.2.2:8000
// For physical device: http://YOUR_COMPUTER_IP:8000
const API_BASE_URL = __DEV__ 
  ? 'http://localhost:8000/api/v1'
  : 'https://your-production-api.com/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface WeatherInput {
  date: string;
  temp_max: number;
  temp_min: number;
  precipitation: number;
  wind_speed: number;
}

export interface AllergenScore {
  allergen_type: string;
  severity: number;
  severity_level: string;
  percentage: number;
}

export interface PredictionResponse {
  date: string;
  total_pollen: number;
  severity_level: string;
  severity_score: number;
  allergen_breakdown: AllergenScore[];
  weather_conditions: {
    temp_max: number;
    temp_min: number;
    temp_avg: number;
    precipitation: number;
    wind_speed: number;
  };
  recommendation: string;
  confidence: number;
}

export interface WeeklyPrediction {
  date: string;
  total_pollen: number;
  severity_level: string;
  severity_score: number;
}

export interface WeeklyPredictionResponse {
  start_date: string;
  end_date: string;
  predictions: WeeklyPrediction[];
  trend: string;
  average_severity: number;
}

export const pollenApi = {
  /**
   * Get daily pollen prediction
   */
  async getDailyPrediction(weather: WeatherInput): Promise<PredictionResponse> {
    const response = await api.post('/predict/daily', weather);
    return response.data;
  },

  /**
   * Get weekly pollen predictions
   */
  async getWeeklyPrediction(weatherList: WeatherInput[]): Promise<WeeklyPredictionResponse> {
    const response = await api.post('/predict/weekly', { weather_data: weatherList });
    return response.data;
  },

  /**
   * Identify dominant allergens
   */
  async identifyAllergens(weather: WeatherInput): Promise<{
    date: string;
    total_severity: number;
    allergen_breakdown: AllergenScore[];
    dominant_allergen: string;
    alert_level: string;
    recommendations: string[];
  }> {
    const response = await api.post('/allergen/identify', weather);
    return response.data;
  },

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string; message: string }> {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api;
