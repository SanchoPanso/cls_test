import os
import sys
from pathlib import Path

from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import create_engine

from cls.classification.utils.cfg import get_cfg

cfg = get_cfg()
DEFAULT_PICTURES_INFO_DB_PATH = cfg['pictures_info_db_path']
ENGINES = {}

class Base(DeclarativeBase):
    pass

class Picture(Base):
    __tablename__ = "pictures"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String(255))
    model_version: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(255))


    def __repr__(self) -> str:
        return f"Picture(id={self.id!r}, path={self.path!r}, model_version={self.model_version!r}, status={self.status!r})"


def get_db_engine(db_path: str = DEFAULT_PICTURES_INFO_DB_PATH):
    if db_path not in ENGINES:
        engine = create_engine(f"sqlite:///{cfg['pictures_info_db_path']}", echo=False)
        Base.metadata.create_all(engine)
        ENGINES[db_path] = engine
    
    engine = ENGINES[db_path]
    return engine


def select_picture_by_path(path: str, db_path: str = DEFAULT_PICTURES_INFO_DB_PATH) -> Picture:
    engine = get_db_engine(db_path)
    session = Session(engine)
    stmt = select(Picture).where(Picture.path == path)
    picture = session.scalars(stmt).one_or_none()
    return picture

def select_picture_by_paths(paths: List[str], db_path: str = DEFAULT_PICTURES_INFO_DB_PATH) -> Picture:
    engine = get_db_engine(db_path)
    with Session(engine) as session:
        stmt = select(Picture).where(Picture.path.in_(paths))
        res = [picture for picture in session.scalars(stmt)]
    return res

def select_picture_by_status(status: List[str], db_path: str = DEFAULT_PICTURES_INFO_DB_PATH) -> Picture:
    engine = get_db_engine(db_path)
    with Session(engine) as session:
        stmt = select(Picture).where(Picture.status == status)
        res = [picture for picture in session.scalars(stmt)]
    return res


def get_rejected_paths(db_path: str = DEFAULT_PICTURES_INFO_DB_PATH):
    pictures = select_picture_by_status('rejected', db_path)
    paths = [pic.path for pic in pictures]
    return paths

def update_picture_by_path(new_picture: Picture, db_path: str = DEFAULT_PICTURES_INFO_DB_PATH):
    engine = get_db_engine(db_path)
    
    with Session(engine) as session:
        stmt = select(Picture).where(Picture.path.in_([new_picture.path]))
        picture = session.scalars(stmt).one_or_none()
    
        if picture is None:
            session.add_all([new_picture])
        else:
            picture.model_version = new_picture.model_version
            picture.status = new_picture.status

        session.commit()


# update_picture_by_path(Picture(path='111.jpg', model_version='v2', status='approved'))
# print(select_picture_by_path('111.jpg'))
