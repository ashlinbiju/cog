# Google Cloud Free Tier API Setup Guide

## Step 1: Google Cloud Console Setup

1. **Go to Google Cloud Console**: https://console.cloud.google.com/
2. **Create/Select Project**:
   - Click "Select a project" → "New Project"
   - Name: "hotel-recommendations" 
   - Click "Create"

## Step 2: Enable Billing (Required but FREE)

1. Go to **Billing** in the left menu
2. **Link a billing account** (required even for free tier)
3. **Add payment method** (won't be charged within free limits)
4. **Important**: You get $300 free credit for 90 days

## Step 3: Enable Required APIs (FREE TIER)

Go to **APIs & Services → Library** and search/enable:

### ✅ **Geocoding API** 
- **Free Quota**: 40,000 requests/month
- **Usage**: Convert addresses to coordinates
- **Enable**: Search "Geocoding API" → Click → Enable

### ✅ **Places API (New)**
- **Free Credit**: $200/month (covers ~11,000 requests)
- **Usage**: Find hotels and get details
- **Enable**: Search "Places API (New)" → Click → Enable

### ✅ **Maps JavaScript API**
- **Free Quota**: Unlimited for websites
- **Usage**: Display maps on frontend
- **Enable**: Search "Maps JavaScript API" → Click → Enable

## Step 4: Create API Key

1. Go to **APIs & Services → Credentials**
2. Click **+ Create Credentials → API Key**
3. **Copy the API key** (starts with AIza...)
4. **For now, don't restrict it** (for testing)

## Step 5: Update Your .env File

Replace your current API key in `.env`:
```
GOOGLE_MAPS_API_KEY=YOUR_NEW_API_KEY_HERE
```

## Step 6: Test Your Setup

Visit these URLs after restarting your app:
- http://127.0.0.1:5000/api/status/status
- http://127.0.0.1:5000/api/status/google-api-test

## Free Tier Limits (Monthly)

| API | Free Quota | Estimated Usage |
|-----|------------|-----------------|
| Geocoding | 40,000 requests | ~1,300 searches/day |
| Places API | $200 credit (~11,000 requests) | ~350 searches/day |
| Maps JavaScript | Unlimited | No limit |

## Cost Optimization Tips

1. **Cache results** - Don't repeat same location searches
2. **Use fallback system** - Your smart recommendations work without Google
3. **Limit requests** - Set daily limits in Google Cloud Console
4. **Monitor usage** - Check APIs & Services → Quotas

## Troubleshooting

### "REQUEST_DENIED" Error:
- API not enabled in Google Cloud Console
- Billing not set up
- API key restrictions too strict

### "OVER_QUERY_LIMIT" Error:
- Exceeded free quota
- Need to wait for quota reset
- Consider upgrading or using fallback

### "INVALID_REQUEST" Error:
- Check location format
- Verify API key is correct

## Security (Optional - After Testing)

Once working, restrict your API key:
1. Go to **APIs & Services → Credentials**
2. Click your API key
3. **Application restrictions**: HTTP referrers
4. **Add**: `127.0.0.1:5000/*`, `localhost:5000/*`
5. **API restrictions**: Select only the 3 APIs above

## Monitoring Usage

1. Go to **APIs & Services → Quotas**
2. Monitor daily usage
3. Set up alerts if approaching limits
4. View detailed usage in **APIs & Services → Dashboard**

Your system will work perfectly within free limits for development and testing!
