from datetime import datetime, timedelta
from pydantic import BaseModel,Field,computed_field
from typing import Optional

date_str = "2026-12-28"
days = 5

final_date = (
    datetime.strptime(date_str, "%Y-%m-%d")
    + timedelta(days=days)
).date()


class date(BaseModel):
    initial:Optional[str]=None
    days:Optional[int]=None

    @computed_field
    @property
    def final_date(self)->Optional[str]:

        if self.initial is None or self.days is None:
            return None
        final_d=(
                datetime.strptime(self.initial, "%Y-%m-%d")
                + timedelta(days=self.days)
                ).strftime("%Y-%m-%d")
        return final_d


input={"initial":"2026-06-06"}
print(date(**input))
