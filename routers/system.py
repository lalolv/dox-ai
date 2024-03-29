from fastapi import APIRouter


router = APIRouter(prefix="/sys")


@router.get("/info")
async def get_sys_info():
    return {
        "status": "ok",
        "version": "0.1.0",
        "author": "Lalo"
    }
