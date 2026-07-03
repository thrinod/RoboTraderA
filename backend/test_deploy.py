
import asyncio
import sys
sys.path.append('i:/RoboTrader/backend')
from main import list_deployments, app
from motor.motor_asyncio import AsyncIOMotorClient
from app.services.telegram_service import telegram_service
from app.services.upstox_service import upstox_service

async def test():
    mongo_client = AsyncIOMotorClient('mongodb://localhost:27017')
    app.mongodb = mongo_client['robotrader']
    telegram_service.set_db(app.mongodb)
    await upstox_service.load_token(app.mongodb)
    
    result = await list_deployments()
    for d in result['data']:
        print('Deploy:', d.get('_id'), 'Name:', d.get('instrument_name'), 'Strike:', d.get('instrument_strike'), 'PK:', d.get('primary_instrument'))

asyncio.run(test())

