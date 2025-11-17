# Pollen Predictor - Mobile App

React Native mobile application for predicting pollen levels using weather data.

## Features

- 🌤️ **Weather Input Form** - Enter local weather conditions
- 📊 **Pollen Predictions** - Get AI-powered pollen severity predictions (0-10 scale)
- 🌿 **Allergen Breakdown** - View detailed breakdown by allergen type (Tree, Grass, Ragweed, Weed)
- 📱 **Apple Design** - Beautiful iOS-inspired glassmorphism design
- 🌓 **Dark Mode** - Full dark mode support

## Prerequisites

- Node.js >= 20.15.0
- npm or yarn
- Expo CLI
- Backend API running (see ../backend/README.md)

## Installation

```bash
npm install
```

## Configuration

Update the API base URL in `services/api.ts` based on your environment:

```typescript
// For iOS simulator
const API_BASE_URL = 'http://localhost:8000/api/v1'

// For Android emulator
const API_BASE_URL = 'http://10.0.2.2:8000/api/v1'

// For physical device (replace with your computer's IP)
const API_BASE_URL = 'http://192.168.1.XXX:8000/api/v1'
```

## Running the App

1. **Start the backend API server first** (from backend directory):
   ```bash
   cd ../backend
   uvicorn app.main:app --reload
   ```

2. **Start the Expo development server**:
   ```bash
   npx expo start
   ```

3. **Choose your platform**:
   - Press `i` for iOS simulator
   - Press `a` for Android emulator
   - Scan QR code with Expo Go app on physical device

## Usage

### Making a Prediction

1. Navigate to **Predict** tab
2. Enter weather data:
   - Maximum temperature (°F)
   - Minimum temperature (°F)
   - Precipitation (inches)
   - Wind speed (mph)
3. Tap **"Get Prediction"**
4. View results on **Results** tab (auto-navigates)

### Understanding Results

**Severity Score** (0-10):
- 0-2: Very Low
- 2-4: Low  
- 4-6: Moderate
- 6-8: High
- 8-10: Very High

**Allergen Breakdown**: Tree, Grass, Ragweed, Weed contributions  
**Weather Conditions**: Input data used for prediction  
**Health Advisory**: Personalized recommendations

## Troubleshooting

### Cannot connect to backend
- Verify backend is running at `http://localhost:8000`
- Check API URL in `services/api.ts`
- For Android emulator, use `10.0.2.2` instead of `localhost`
- For physical device, use your computer's IP on same network

### "Network request failed"
- Backend may not be running
- Firewall blocking connection
- Wrong API URL configuration

## Technologies

- React Native + Expo
- TypeScript
- Expo Router (file-based navigation)
- React Native Reanimated (animations)
- Axios (HTTP client)
- Expo Blur (glassmorphism)
- Linear Gradient (backgrounds)

## Get a fresh project

When you're ready, run:

```bash
npm run reset-project
```

This command will move the starter code to the **app-example** directory and create a blank **app** directory where you can start developing.

## Learn more

To learn more about developing your project with Expo, look at the following resources:

- [Expo documentation](https://docs.expo.dev/): Learn fundamentals, or go into advanced topics with our [guides](https://docs.expo.dev/guides).
- [Learn Expo tutorial](https://docs.expo.dev/tutorial/introduction/): Follow a step-by-step tutorial where you'll create a project that runs on Android, iOS, and the web.

## Join the community

Join our community of developers creating universal apps.

- [Expo on GitHub](https://github.com/expo/expo): View our open source platform and contribute.
- [Discord community](https://chat.expo.dev): Chat with Expo users and ask questions.
