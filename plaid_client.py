import os
from plaid import Client
from plaid.api import plaid_api
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
PLAID_SECRET = os.getenv("PLAID_SECRET")
PLAID_ENV = os.getenv("PLAID_ENV", "sandbox")

configuration = plaid_api.Configuration(
    host=plaid_api.Environment.Sandbox if PLAID_ENV == "sandbox" else plaid_api.Environment.Production,
    api_key={
        'clientId': PLAID_CLIENT_ID,
        'secret': PLAID_SECRET,
    }
)
api_client = plaid_api.ApiClient(configuration)
client = plaid_api.PlaidApi(api_client)

def create_link_token(user_id: str):
    request = LinkTokenCreateRequest(
        products=[Products("transactions")],
        client_name="Subscription Leakage Intelligence",
        country_codes=[CountryCode('US')],
        language="en",
        user={"client_user_id": user_id},
    )
    response = client.link_token_create(request)
    return response.to_dict()

def exchange_public_token(public_token: str):
    request = ItemPublicTokenExchangeRequest(public_token=public_token)
    response = client.item_public_token_exchange(request)
    return response.to_dict()

def fetch_transactions(access_token: str, start_date: str, end_date: str):
    request = TransactionsGetRequest(
        access_token=access_token,
        start_date=start_date,
        end_date=end_date,
        options=TransactionsGetRequestOptions(count=500, offset=0)
    )
    response = client.transactions_get(request)
    return response.to_dict()
