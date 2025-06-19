import logging
from datetime import datetime, timedelta
from typing import Optional

import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, status, Header
from pydantic import BaseModel
import requests

from .config import global_args

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class TokenPayload(BaseModel):
    sub: str  # Username
    exp: datetime  # Expiration time
    role: str = "user"  # User role, default is regular user
    metadata: dict = {}  # Additional metadata


class AuthHandler:
    def __init__(self):
        self.secret = global_args.token_secret
        self.algorithm = global_args.jwt_algorithm
        self.expire_hours = global_args.token_expire_hours
        self.guest_expire_hours = global_args.guest_token_expire_hours
        self.accounts = {}
        auth_accounts = global_args.auth_accounts
        if auth_accounts:
            for account in auth_accounts.split(","):
                username, password = account.split(":", 1)
                self.accounts[username] = password

    def create_token(
        self,
        username: str,
        role: str = "user",
        custom_expire_hours: int = None,
        metadata: dict = None,
    ) -> str:
        """
        Create JWT token

        Args:
            username: Username
            role: User role, default is "user", guest is "guest"
            custom_expire_hours: Custom expiration time (hours), if None use default value
            metadata: Additional metadata

        Returns:
            str: Encoded JWT token
        """
        # Choose default expiration time based on role
        if custom_expire_hours is None:
            if role == "guest":
                expire_hours = self.guest_expire_hours
            else:
                expire_hours = self.expire_hours
        else:
            expire_hours = custom_expire_hours

        expire = datetime.utcnow() + timedelta(hours=expire_hours)

        # Create payload
        payload = TokenPayload(
            sub=username, exp=expire, role=role, metadata=metadata or {}
        )

        return jwt.encode(payload.dict(), self.secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> dict:
        """
        Validate JWT token

        Args:
            token: JWT token

        Returns:
            dict: Dictionary containing user information

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            expire_timestamp = payload["exp"]
            expire_time = datetime.utcfromtimestamp(expire_timestamp)

            if datetime.utcnow() > expire_time:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
                )

            # Return complete payload instead of just username
            return {
                "username": payload["sub"],
                "role": payload.get("role", "user"),
                "metadata": payload.get("metadata", {}),
                "exp": expire_time,
            }
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )


auth_handler = AuthHandler()


# !todo: Implement the login function
async def get_current_user(Authorization: Optional[str] = Header(None, description="Authorization token")) -> str:
    """
    Get the current user from the token.

    Args:
        Authorization: token

    Returns:
        dict: User information including username, role, and metadata

    Raises:
        HTTPException: If token is invalid or expired
    """
    # auth_service_url = "https://your-auth-backend.com/validate_token"
    # headers = {"Authorization": f"Bearer {token}"}
    #
    # try:
    #     response = requests.post(auth_service_url, headers=headers)
    #     response.raise_for_status()  # Raises an exception for 4XX/5XX status
    #
    #     user_data = response.json()
    #     user_id = user_data.get("user_id")
    #
    #     if not user_id:
    #         raise HTTPException(
    #             status_code=status.HTTP_401_UNAUTHORIZED,
    #             detail="Invalid token or user_id missing",
    #         )
    #     return user_id
    # except requests.RequestException as e:
    #     raise HTTPException(
    #         status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    #         detail=f"Authentication service unavailable: {e}",
    #     )
    # except Exception:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Could not validate credentials",
    #         headers={"WWW-Authenticate": "Bearer"},
    #     )

    return "test_user"


async def mock_get_current_user_id(x_user_id: Optional[str] = Header(None, description="用于测试的用户ID")) -> str:
    """
    一个模拟的认证依赖项。
    它从请求头 'X-User-ID' 中获取用户ID。
    如果请求头不存在，它会返回一个默认的测试用户ID。
    这允许我们在没有真实认证系统的情况下测试多租户功能。
    """
    if x_user_id:
        logging.info(f"Simulating user login for: {x_user_id}")
        return x_user_id
    # 在生产环境中，如果令牌无效或缺失，您应该抛出 HTTPException(status_code=401)
    logging.warning("X-User-ID header not found, using default 'test_user' for simulation.")
    return "test_user"