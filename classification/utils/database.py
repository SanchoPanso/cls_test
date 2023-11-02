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

sys.path.append(str(Path(__file__).parent.parent))
from utils.cfg import get_cfg



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


cfg = get_cfg()
data_path = cfg['data_path']
ENGINE = create_engine(f"sqlite:///{os.path.join(data_path, 'file.db')}", echo=False)
Base.metadata.create_all(ENGINE)


def select_picture_by_path(path: str) -> Picture:
    session = Session(ENGINE)
    stmt = select(Picture).where(Picture.path.in_([path]))
    picture = session.scalars(stmt).one_or_none()
    return picture


def update_picture_by_path(new_picture: Picture):
    
    with Session(ENGINE) as session:
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
