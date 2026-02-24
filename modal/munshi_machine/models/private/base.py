from typing import Any, Dict, Iterable, Optional, Type, TypeVar, List, Tuple

from pydantic import ValidationError
from sqlmodel import MetaData, SQLModel, select, Session
from sqlmodel.ext.asyncio.session import AsyncSession

T = TypeVar("T", bound="PrivateSchemaBase")

class PrivateSchemaBase(SQLModel):
    metadata = MetaData(schema="private")

    @classmethod
    async def find_replace_async(
        cls: Type[T],
        session: AsyncSession,
        find_by: Dict[str, Any],
        replace_with: Dict[str, Any],
        upsert: bool = True,
        commit: bool = True,
    ) -> Optional[T]:
        """
        Async find-replace with upsert, plus strict field & type validation.
        """

        # ---- 1) FIELD VALIDATION ------------------------------------------------
        model_fields = cls.model_fields  # SQLModel inherits from Pydantic v2

        unknown_find = set(find_by.keys()) - set(model_fields.keys())
        unknown_replace = set(replace_with.keys()) - set(model_fields.keys())

        if unknown_find:
            raise ValueError(f"Unknown fields in find_by: {unknown_find}")

        if unknown_replace:
            raise ValueError(f"Unknown fields in replace_with: {unknown_replace}")

        # Validate types early using SQLModel's pydantic model
        # try:
        #     cls.model_validate(find_by, from_attributes=False)
        #     cls.model_validate(replace_with, from_attributes=False)
        # except ValidationError as e:
        #     raise ValueError(f"Validation failed: {e}") from e

        # ---- 2) BUILD QUERY -----------------------------------------------------
        query = select(cls)
        for field, value in find_by.items():
            query = query.where(getattr(cls, field) == value)

        result = (await session.exec(query)).first()

        # ---- 3) FOUND → UPDATE --------------------------------------------------
        if result:
            for field, value in replace_with.items():
                setattr(result, field, value)
            # session.add(result)
            if commit:
                await session.commit()
                await session.refresh(result)
            return result

        # ---- 4) NOT FOUND → UPSERT ---------------------------------------------
        if upsert:
            data = {**find_by, **replace_with}

            # Validate entire object before creation
            try:
                obj = cls.model_validate(data)
            except ValidationError as e:
                raise ValueError(
                    f"Validation failed during object creation: {e}"
                ) from e

            new_obj: T = cls(**obj.model_dump())
            session.add(new_obj)
            if commit:
                await session.commit()
                await session.refresh(new_obj)
            return new_obj

        # ---- 5) NOT FOUND AND NO UPSERT ----------------------------------------
        return None
    

    @classmethod
    async def batch_find_replace_async(
        cls: Type[T],
        session: AsyncSession,
        operations: Iterable[
            Tuple[
                Dict[str, Any],   # find_by
                Dict[str, Any],   # replace_with
            ]
        ],
        upsert: bool = True,
    ) -> List[T]:
        """
        Batch version of find_replace_async.

        - Executes all operations with commit=False
        - Commits once at the end
        - Refreshes each object after commit
        - Returns results in the same order as operations
        """

        results: List[T] = []

        # -- Phase 1: Execute operations without committing
        for find_by, replace_with in operations:
            obj = await cls.find_replace_async(
                session,
                find_by=find_by,
                replace_with=replace_with,
                upsert=upsert,
                commit=False,   # batch mode
            )
            results.append(obj)

        # -- Phase 2: Commit once
        await session.commit()

        # -- Phase 3: Refresh every returned object
        for obj in results:
            if obj is not None:
                await session.refresh(obj)

        return results

    @classmethod
    async def fast_batch_find_replace_async(
        cls,
        session: AsyncSession,
        operations: list[tuple[dict, dict]],
        upsert=True,
        refresh_new=True,
    ):
        # Extract unique lookup keys (assume 1-field find_by)
        lookup_values = [list(find_by.values())[0] for find_by, _ in operations]
        lookup_field = list(operations[0][0].keys())[0]

        # 1) Batch-load all existing objects in one query
        stmt = select(cls).where(getattr(cls, lookup_field).in_(lookup_values))
        existing_objs = (await session.exec(stmt)).all()

        existing_map = {getattr(obj, lookup_field): obj for obj in existing_objs}

        results = []
        new_objects = []
        modified = 0

        # 2) Apply updates or build new objects
        for find_by, replace_with in operations:
            value = list(find_by.values())[0]

            if value in existing_map:
                obj = existing_map[value]
                for k, v in replace_with.items():
                    setattr(obj, k, v)
                results.append(obj)
                modified += 1
            else:
                if upsert:
                    data = {**find_by, **replace_with}
                    new_obj = cls(**data)
                    new_objects.append(new_obj)
                    results.append(new_obj)
                else:
                    results.append(None)

        # 3) Batch add new objects (one round-trip)
        if new_objects:
            session.add_all(new_objects)

        # 4) Commit once
        await session.commit()

        # 5) Refresh only newly inserted objects
        if refresh_new:
            for obj in new_objects:
                await session.refresh(obj)

        return results, modified

    @classmethod
    def find_replace_sync(
        cls: Type[T],
        session: Session,
        find_by: Dict[str, Any],
        replace_with: Dict[str, Any],
        upsert: bool = True,
        commit: bool = True,
    ) -> Optional[T]:
        """
        Sync find-replace with upsert, plus strict field & type validation.
        """

        # ---- 1) FIELD VALIDATION ------------------------------------------------
        model_fields = cls.model_fields  # SQLModel inherits from Pydantic v2

        unknown_find = set(find_by.keys()) - set(model_fields.keys())
        unknown_replace = set(replace_with.keys()) - set(model_fields.keys())

        if unknown_find:
            raise ValueError(f"Unknown fields in find_by: {unknown_find}")

        if unknown_replace:
            raise ValueError(f"Unknown fields in replace_with: {unknown_replace}")

        # ---- 2) BUILD QUERY -----------------------------------------------------
        query = select(cls)
        for field, value in find_by.items():
            query = query.where(getattr(cls, field) == value)

        result = session.exec(query).first()

        # ---- 3) FOUND → UPDATE --------------------------------------------------
        if result:
            for field, value in replace_with.items():
                setattr(result, field, value)
            if commit:
                session.commit()
                session.refresh(result)
            return result

        # ---- 4) NOT FOUND → UPSERT ---------------------------------------------
        if upsert:
            data = {**find_by, **replace_with}

            # Validate entire object before creation
            try:
                obj = cls.model_validate(data)
            except ValidationError as e:
                raise ValueError(
                    f"Validation failed during object creation: {e}"
                ) from e

            new_obj: T = cls(**obj.model_dump())
            session.add(new_obj)
            if commit:
                session.commit()
                session.refresh(new_obj)
            return new_obj

        # ---- 5) NOT FOUND AND NO UPSERT ----------------------------------------
        return None
    

    @classmethod
    def batch_find_replace_sync(
        cls: Type[T],
        session: Session,
        operations: Iterable[
            Tuple[
                Dict[str, Any],   # find_by
                Dict[str, Any],   # replace_with
            ]
        ],
        upsert: bool = True,
    ) -> List[T]:
        """
        Batch version of find_replace_sync.

        - Executes all operations with commit=False
        - Commits once at the end
        - Refreshes each object after commit
        - Returns results in the same order as operations
        """

        results: List[T] = []

        # -- Phase 1: Execute operations without committing
        for find_by, replace_with in operations:
            obj = cls.find_replace_sync(
                session,
                find_by=find_by,
                replace_with=replace_with,
                upsert=upsert,
                commit=False,   # batch mode
            )
            results.append(obj)

        # -- Phase 2: Commit once
        session.commit()

        # -- Phase 3: Refresh every returned object
        for obj in results:
            if obj is not None:
                session.refresh(obj)

        return results

    @classmethod
    def fast_batch_find_replace_sync(
        cls,
        session: Session,
        operations: list[tuple[dict, dict]],
        upsert=True,
        refresh_new=True,
    ):
        # Extract unique lookup keys (assume 1-field find_by)
        lookup_values = [list(find_by.values())[0] for find_by, _ in operations]
        lookup_field = list(operations[0][0].keys())[0]

        # 1) Batch-load all existing objects in one query
        stmt = select(cls).where(getattr(cls, lookup_field).in_(lookup_values))
        existing_objs = session.exec(stmt).all()

        existing_map = {getattr(obj, lookup_field): obj for obj in existing_objs}

        results = []
        new_objects = []
        modified = 0

        # 2) Apply updates or build new objects
        for find_by, replace_with in operations:
            value = list(find_by.values())[0]

            if value in existing_map:
                obj = existing_map[value]
                for k, v in replace_with.items():
                    setattr(obj, k, v)
                results.append(obj)
                modified += 1
            else:
                if upsert:
                    data = {**find_by, **replace_with}
                    new_obj = cls(**data)
                    new_objects.append(new_obj)
                    results.append(new_obj)
                else:
                    results.append(None)

        # 3) Batch add new objects (one round-trip)
        if new_objects:
            session.add_all(new_objects)

        # 4) Commit once
        session.commit()

        # 5) Refresh only newly inserted objects
        if refresh_new:
            for obj in new_objects:
                session.refresh(obj)

        return results, modified
