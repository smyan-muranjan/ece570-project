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
  temp_avg?: number;
  precipitation: number;
  wind_speed?: number;
}

export interface AllergenScore {
  allergen_type: string;
  severity_score: number;
  severity_level: string;
  contribution_pct: number;
}

export interface PollenPrediction {
  date: string;
  severity_score: number;
  severity_level: string;
  confidence?: number;
}

export interface PredictionResponse {
  prediction: PollenPrediction;
  recommendation: string;
  allergen_breakdown?: AllergenScore[];
  primary_allergen?: string;
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
   * @param weather - Weather conditions for the prediction day
   * @param historicalPollen - Optional: Last 7 days of raw pollen counts (improves accuracy significantly)
   * @param historicalTemps - Optional: Last 30 days of average temperatures in °F (for anomaly detection)
   * @param historicalPrecip - Optional: Season-to-date precipitation in inches (for cumulative rainfall)
   * @param historicalWind - Optional: Last 30 days of wind speeds in mph (for wind percentile)
   */
  async getDailyPrediction(
    weather: WeatherInput,
    historicalPollen?: number[],
    historicalTemps?: number[],
    historicalPrecip?: number[],
    historicalWind?: number[]
  ): Promise<PredictionResponse> {
    const requestBody: any = { weather };
    if (historicalPollen && historicalPollen.length > 0) {
      requestBody.historical_pollen = historicalPollen;
    }
    if (historicalTemps && historicalTemps.length > 0) {
      requestBody.historical_temps = historicalTemps;
    }
    if (historicalPrecip && historicalPrecip.length > 0) {
      requestBody.historical_precip = historicalPrecip;
    }
    if (historicalWind && historicalWind.length > 0) {
      requestBody.historical_wind = historicalWind;
    }
    const response = await api.post('/predict/daily', requestBody);
    return response.data;
  },

  /**
   * Get weekly pollen predictions
   */
  async getWeeklyPrediction(weatherList: WeatherInput[]): Promise<WeeklyPredictionResponse> {
    const response = await api.post('/predict/weekly', { weather_forecast: weatherList });
    return response.data;
  },

  /**
   * Identify dominant allergens
   * @param weather - Weather conditions for allergen identification
   * @param historicalPollen - Optional: Last 7 days of raw pollen counts
   * @param historicalTemps - Optional: Last 30 days of average temperatures in °F
   * @param historicalPrecip - Optional: Season-to-date precipitation in inches
   * @param historicalWind - Optional: Last 30 days of wind speeds in mph
   */
  async identifyAllergens(
    weather: WeatherInput,
    historicalPollen?: number[],
    historicalTemps?: number[],
    historicalPrecip?: number[],
    historicalWind?: number[]
  ): Promise<{
    date: string;
    total_severity: number;
    allergens: AllergenScore[];
    primary_allergen: string;
    alert_level: string;
  }> {
    const requestBody: any = { weather };
    if (historicalPollen && historicalPollen.length > 0) {
      requestBody.historical_pollen = historicalPollen;
    }
    if (historicalTemps && historicalTemps.length > 0) {
      requestBody.historical_temps = historicalTemps;
    }
    if (historicalPrecip && historicalPrecip.length > 0) {
      requestBody.historical_precip = historicalPrecip;
    }
    if (historicalWind && historicalWind.length > 0) {
      requestBody.historical_wind = historicalWind;
    }
    const response = await api.post('/allergen/identify', requestBody);
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
