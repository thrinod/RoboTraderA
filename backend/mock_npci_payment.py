from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import uuid
import datetime

router = APIRouter()

class MockPaymentRequest(BaseModel):
    transactionId: str
    senderAccount: str
    beneficiaryAccount: str
    beneficiaryIfsc: str
    beneficiaryName: str
    amount: float
    currency: str
    timestamp: str

@router.post("/process-payment")
async def process_mock_payment(requests: List[MockPaymentRequest]):
    """
    Mock NPCI payment endpoint.
    Accepts a list of transaction requests and returns a list of mock responses.
    Can be easily removed later.
    """
    responses = []
    
    for req in requests:
        responses.append({
            "userId": f"USR{uuid.uuid4().hex[:8].upper()}",
            "beneficiaryName": req.beneficiaryName,
            "amount": req.amount,
            "status": "SUCCESS",
            "transactionId": req.transactionId,
            "npciTransactionId": f"NPCI{uuid.uuid4().hex[:12].upper()}",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
        
    return responses
