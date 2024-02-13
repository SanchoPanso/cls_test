import psycopg2
import json
from typing import List
from sqlalchemy import String, JSON
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import create_engine, Engine


class Base(DeclarativeBase):
    pass


class Picture(Base):
    __tablename__ = "pictures"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String(255))
    model_version: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(255))
    segments: Mapped[str] = mapped_column(JSON())
    

    def __repr__(self) -> str:
        repr =  f"Picture(id={self.id!r}, " \
                f"path={self.path!r}, "     \
                f"model_version={self.model_version!r}, "   \
                f"status={self.status!r}, "  \
                f"segments={self.segments.keys()!r})"

        return repr


class PostgreSQLHandler:
    __shared_engines = {}
    
    def __init__(
        self,
        host="localhost",
        database="psql_db",
        user="psql_user",
        password="root",
        port="5432",
        echo=False,
    ) -> None:
        
        dialect = 'postgresql'
        driver = 'psycopg2'
        url = f'{dialect}+{driver}://{user}:{password}@{host}:{port}/{database}'
        if url not in self.__shared_engines:
            self.__shared_engines[url] = create_engine(url, echo=echo)
        
        self.engine: Engine = self.__shared_engines[url]

    def select_picture_by_path(self, path: str) -> Picture:
        with Session(self.engine) as session:
            stmt = select(Picture).where(Picture.path == path)
            picture = session.scalars(stmt).one_or_none()
        return picture
    
    def select_all_paths(self) -> List[str]:
        with Session(self.engine) as session:
            stmt = select(Picture).where()
            res = [picture.path for picture in session.scalars(stmt)]
        return res
    
    def select_all_pictures(self) -> List[Picture]:
        with Session(self.engine) as session:
            stmt = select(Picture).where()
            res = [picture for picture in session.scalars(stmt)]
        return res
    
    def update_picture_by_path(self, new_picture: Picture):
        with Session(self.engine) as session:
            stmt = select(Picture).where(Picture.path.in_([new_picture.path]))
            picture = session.scalars(stmt).one_or_none()

            if picture is None:
                session.add_all([new_picture])
            else:
                picture.model_version = new_picture.model_version
                picture.status = new_picture.status
                picture.segments = new_picture.segments

            session.commit()

# posgres_handler = PostgreSQLHandler()
# pic = posgres_handler.select_picture_by_path("8c174d21b1e9504483e5128d756f4b0c.jpeg")
# print(pic)

# posgres_handler = PostgreSQLHandler()
# res = posgres_handler.select_all_paths()
# print(res)
